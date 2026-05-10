# Copyright (c) 2025 Kenneth Stott
# Canary: f8919bf2-2a97-491f-ad34-71565b4dcd49
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Audio/video transcription for the document ingestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    start: float  # seconds
    end: float
    text: str
    speaker: str | None = None  # "SPEAKER_00", etc.


@dataclass
class TranscriptionResult:
    segments: list[TranscriptionSegment]
    language: str
    duration: float  # total audio duration in seconds


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_audio(audio_path: Path, doc_config: DocumentConfig) -> TranscriptionResult:
    """Transcribe an audio/video file using faster-whisper, optionally with WhisperX diarization."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is required for audio transcription. "
            "Install with: pip install 'constat[audio]'"
        )

    logger.info("[audio] Loading whisper model %s for %s", doc_config.whisper_model, audio_path.name)
    model = WhisperModel(doc_config.whisper_model, device="auto", compute_type="default")

    logger.info("[audio] Transcribing %s (language=%s, diarize=%s)",
                audio_path.name, doc_config.language or "auto", doc_config.diarize)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=doc_config.language,
        word_timestamps=doc_config.diarize,
    )

    # Consume the generator
    seg_list = list(raw_segments)
    logger.info("[audio] Transcription complete: %d segments, language=%s, duration=%s",
                len(seg_list), info.language, _fmt_timestamp(info.duration))

    segments = [
        TranscriptionSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in seg_list
    ]

    if doc_config.diarize:
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "whisperx is required for speaker diarization. "
                "Install with: pip install 'constat[audio]'"
            )

        if not doc_config.hf_token:
            raise ValueError(
                "hf_token is required for speaker diarization. "
                "Set hf_token in document config or HF_TOKEN environment variable."
            )

        device = "cuda" if _cuda_available() else "cpu"
        logger.info("[audio] Diarization: loading audio on device=%s", device)
        audio = whisperx.load_audio(str(audio_path))

        # Build segment dicts for whisperx alignment
        whisperx_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in seg_list]

        logger.info("[audio] Diarization: aligning %d segments", len(whisperx_segments))
        align_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        aligned = whisperx.align(
            whisperx_segments, align_model, metadata, audio, device
        )

        logger.info("[audio] Diarization: running speaker identification")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=doc_config.hf_token, device=device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, aligned)

        # Update segments with speaker labels
        segments = []
        for s in result["segments"]:
            segments.append(TranscriptionSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"].strip(),
                speaker=s.get("speaker"),
            ))

        speakers = {s.speaker for s in segments if s.speaker}
        logger.info("[audio] Diarization complete: %d segments, %d speakers", len(segments), len(speakers))

    return TranscriptionResult(
        segments=segments,
        language=info.language,
        duration=info.duration,
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def render_transcript(result: TranscriptionResult, name: str) -> str:
    """Render a TranscriptionResult as markdown for chunking."""
    if not result.segments:
        return f"# Transcript: {name}\n\n**Duration:** {_fmt_timestamp(result.duration)} | **Language:** {result.language}\n\n*No speech detected.*"

    has_speakers = any(s.speaker for s in result.segments)
    lines: list[str] = []

    if has_speakers:
        speakers = {s.speaker for s in result.segments if s.speaker}
        lines.append(f"# Transcript: {name}")
        lines.append("")
        lines.append(
            f"**Duration:** {_fmt_timestamp(result.duration)} | "
            f"**Language:** {result.language} | "
            f"**Speakers:** {len(speakers)}"
        )

        # Group consecutive same-speaker segments into turns
        current_speaker = None
        current_texts: list[str] = []
        current_start = 0.0
        for seg in result.segments:
            if seg.speaker != current_speaker:
                if current_texts:
                    lines.append("")
                    lines.append(f"## [{_fmt_timestamp(current_start)}] {current_speaker}")
                    lines.append(" ".join(current_texts))
                current_speaker = seg.speaker
                current_texts = [seg.text]
                current_start = seg.start
            else:
                current_texts.append(seg.text)
        # Final turn
        if current_texts:
            lines.append("")
            lines.append(f"## [{_fmt_timestamp(current_start)}] {current_speaker}")
            lines.append(" ".join(current_texts))
    else:
        lines.append(f"# Transcript: {name}")
        lines.append("")
        lines.append(
            f"**Duration:** {_fmt_timestamp(result.duration)} | "
            f"**Language:** {result.language}"
        )
        for seg in result.segments:
            lines.append("")
            lines.append(f"[{_fmt_timestamp(seg.start)}] {seg.text}")

    return "\n".join(lines)
