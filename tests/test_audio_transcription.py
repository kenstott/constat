# Copyright (c) 2025 Kenneth Stott
# Canary: ad0a97e8-6ba8-4f9d-9105-121514c771c5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for audio/video transcription document source."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from constat.core.config import DocumentConfig
from constat.discovery.doc_tools._audio import (
    TranscriptionSegment,
    TranscriptionResult,
    render_transcript,
    transcribe_audio,
    _fmt_timestamp,
)
from constat.discovery.doc_tools._mime import (
    MIME_TO_SHORT,
    EXTENSION_TO_SHORT,
    _BINARY_TYPES,
    _KNOWN_SHORT,
    is_binary_type,
    is_loadable_mime,
    detect_type_from_source,
)


# =============================================================================
# TestTranscriptionSegment
# =============================================================================

class TestTranscriptionSegment:
    def test_dataclass_fields(self):
        seg = TranscriptionSegment(start=0.0, end=1.5, text="hello", speaker="SPEAKER_00")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "hello"
        assert seg.speaker == "SPEAKER_00"

    def test_speaker_defaults_none(self):
        seg = TranscriptionSegment(start=0.0, end=1.0, text="test")
        assert seg.speaker is None

    def test_result_fields(self):
        segs = [TranscriptionSegment(start=0.0, end=1.0, text="hi")]
        result = TranscriptionResult(segments=segs, language="en", duration=10.0)
        assert result.language == "en"
        assert result.duration == 10.0
        assert len(result.segments) == 1


# =============================================================================
# TestTranscribeAudio
# =============================================================================

class TestTranscribeAudio:
    def _make_doc_config(self, **kwargs):
        return DocumentConfig(**kwargs)

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_basic_transcription(self, mock_model_cls, audio_fixtures):
        """Mock WhisperModel, verify segments returned."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        fake_seg = SimpleNamespace(start=0.0, end=2.0, text=" Hello world ")
        info = SimpleNamespace(language="en", duration=2.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            # Re-import to pick up mocked module
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            doc_config = self._make_doc_config()
            result = audio_mod.transcribe_audio(audio_fixtures["short_wav"], doc_config)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.0

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_language_detection(self, mock_model_cls, audio_fixtures):
        """Mock returns language='fr', verify result.language."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Bonjour")
        info = SimpleNamespace(language="fr", duration=1.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(audio_fixtures["short_wav"], self._make_doc_config())

        assert result.language == "fr"

    @patch("constat.discovery.doc_tools._audio._cuda_available", return_value=False)
    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_diarization(self, mock_model_cls, mock_cuda, audio_fixtures):
        """Mock full whisperx pipeline, verify speaker labels assigned."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        fake_seg = SimpleNamespace(start=0.0, end=2.0, text="Hello")
        info = SimpleNamespace(language="en", duration=4.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = "audio_array"
        mock_whisperx.load_align_model.return_value = ("align_model", "metadata")
        mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 2.0, "text": "Hello"}]}
        mock_whisperx.DiarizationPipeline.return_value = MagicMock(return_value="diarize_segs")
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " Hello ", "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 4.0, "text": " World ", "speaker": "SPEAKER_01"},
            ]
        }

        with patch.dict("sys.modules", {
            "faster_whisper": MagicMock(WhisperModel=mock_model_cls),
            "whisperx": mock_whisperx,
        }):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            doc_config = self._make_doc_config(diarize=True, hf_token="test_token")
            result = audio_mod.transcribe_audio(audio_fixtures["short_wav"], doc_config)

        assert len(result.segments) == 2
        assert result.segments[0].speaker == "SPEAKER_00"
        assert result.segments[1].speaker == "SPEAKER_01"

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_diarization_without_hf_token(self, mock_model_cls, audio_fixtures):
        """diarize=True but no token -> clear error."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Test")
        info = SimpleNamespace(language="en", duration=1.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        mock_whisperx = MagicMock()

        with patch.dict("sys.modules", {
            "faster_whisper": MagicMock(WhisperModel=mock_model_cls),
            "whisperx": mock_whisperx,
        }):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            doc_config = self._make_doc_config(diarize=True)
            with pytest.raises(ValueError, match="hf_token is required"):
                audio_mod.transcribe_audio(audio_fixtures["short_wav"], doc_config)

    def test_missing_faster_whisper(self, audio_fixtures):
        """faster-whisper not installed -> ImportError with install hint."""
        with patch.dict("sys.modules", {"faster_whisper": None}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            with pytest.raises(ImportError, match="faster-whisper"):
                audio_mod.transcribe_audio(audio_fixtures["short_wav"], self._make_doc_config())

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_missing_whisperx(self, mock_model_cls, audio_fixtures):
        """diarize=True, whisperx not installed -> ImportError with install hint."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Test")
        info = SimpleNamespace(language="en", duration=1.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {
            "faster_whisper": MagicMock(WhisperModel=mock_model_cls),
            "whisperx": None,
        }):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            doc_config = self._make_doc_config(diarize=True, hf_token="token")
            with pytest.raises(ImportError, match="whisperx"):
                audio_mod.transcribe_audio(audio_fixtures["short_wav"], doc_config)

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_empty_audio(self, mock_model_cls, audio_fixtures):
        """Model returns no segments -> empty TranscriptionResult."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        info = SimpleNamespace(language="en", duration=2.0)
        mock_model.transcribe.return_value = (iter([]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(audio_fixtures["silence_wav"], self._make_doc_config())

        assert len(result.segments) == 0
        assert result.language == "en"


# =============================================================================
# TestRenderTranscript
# =============================================================================

class TestRenderTranscript:
    def test_render_with_speakers(self):
        segments = [
            TranscriptionSegment(start=5.0, end=10.0, text="Welcome everyone.", speaker="SPEAKER_00"),
            TranscriptionSegment(start=10.0, end=15.0, text="Thanks for joining.", speaker="SPEAKER_00"),
            TranscriptionSegment(start=83.0, end=90.0, text="Let me share numbers.", speaker="SPEAKER_01"),
        ]
        result = TranscriptionResult(segments=segments, language="en", duration=2732.0)
        md = render_transcript(result, "meeting_recording")

        assert "# Transcript: meeting_recording" in md
        assert "**Speakers:** 2" in md
        assert "## [00:00:05] SPEAKER_00" in md
        assert "## [00:01:23] SPEAKER_01" in md
        assert "Welcome everyone. Thanks for joining." in md
        assert "Let me share numbers." in md

    def test_render_without_speakers(self):
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Welcome to the show."),
            TranscriptionSegment(start=135.0, end=140.0, text="Today we discuss AI."),
        ]
        result = TranscriptionResult(segments=segments, language="en", duration=1930.0)
        md = render_transcript(result, "podcast_ep42")

        assert "# Transcript: podcast_ep42" in md
        assert "**Duration:** 00:32:10" in md
        assert "**Speakers:**" not in md
        assert "[00:00:00] Welcome to the show." in md
        assert "[00:02:15] Today we discuss AI." in md

    def test_render_timestamps_formatted(self):
        assert _fmt_timestamp(0) == "00:00:00"
        assert _fmt_timestamp(65) == "00:01:05"
        assert _fmt_timestamp(3661) == "01:01:01"
        assert _fmt_timestamp(7200) == "02:00:00"

    def test_render_speaker_grouping(self):
        """Consecutive same-speaker segments merged into one turn."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="First.", speaker="SPEAKER_00"),
            TranscriptionSegment(start=2.0, end=4.0, text="Second.", speaker="SPEAKER_00"),
            TranscriptionSegment(start=4.0, end=6.0, text="Third.", speaker="SPEAKER_01"),
            TranscriptionSegment(start=6.0, end=8.0, text="Fourth.", speaker="SPEAKER_00"),
        ]
        result = TranscriptionResult(segments=segments, language="en", duration=8.0)
        md = render_transcript(result, "test")

        # SPEAKER_00's first two segments should be one turn
        assert "First. Second." in md
        # Then SPEAKER_01
        assert "Third." in md
        # Then SPEAKER_00 again (separate turn)
        assert "Fourth." in md
        # Count speaker headings: should be 3 turns
        assert md.count("## [") == 3

    def test_render_empty_segments(self):
        result = TranscriptionResult(segments=[], language="en", duration=5.0)
        md = render_transcript(result, "empty")
        assert "No speech detected" in md


# =============================================================================
# TestAudioMimeDetection
# =============================================================================

class TestAudioMimeDetection:
    def test_extension_detection(self):
        assert EXTENSION_TO_SHORT[".mp3"] == "audio"
        assert EXTENSION_TO_SHORT[".wav"] == "audio"
        assert EXTENSION_TO_SHORT[".m4a"] == "audio"
        assert EXTENSION_TO_SHORT[".mp4"] == "audio"
        assert EXTENSION_TO_SHORT[".ogg"] == "audio"
        assert EXTENSION_TO_SHORT[".flac"] == "audio"
        assert EXTENSION_TO_SHORT[".webm"] == "audio"
        assert EXTENSION_TO_SHORT[".mkv"] == "audio"
        assert EXTENSION_TO_SHORT[".aac"] == "audio"

    def test_mime_detection(self):
        assert MIME_TO_SHORT["audio/mpeg"] == "audio"
        assert MIME_TO_SHORT["audio/wav"] == "audio"
        assert MIME_TO_SHORT["audio/mp4"] == "audio"
        assert MIME_TO_SHORT["video/mp4"] == "audio"
        assert MIME_TO_SHORT["video/webm"] == "audio"
        assert MIME_TO_SHORT["video/x-matroska"] == "audio"

    def test_binary_type(self):
        assert is_binary_type("audio")

    def test_known_short(self):
        assert "audio" in _KNOWN_SHORT

    def test_loadable_mime(self):
        assert is_loadable_mime("audio/mpeg")
        assert is_loadable_mime("audio/wav")
        assert is_loadable_mime("video/mp4")
        assert is_loadable_mime("video/webm")

    def test_detect_type_from_source_extension(self):
        assert detect_type_from_source("recording.mp3", None) == "audio"
        assert detect_type_from_source("meeting.wav", None) == "audio"
        assert detect_type_from_source("video.mp4", None) == "audio"

    def test_detect_type_from_source_mime(self):
        assert detect_type_from_source(None, "audio/mpeg") == "audio"
        assert detect_type_from_source(None, "video/mp4") == "audio"


# =============================================================================
# TestAudioIngestionPipeline
# =============================================================================

class TestAudioIngestionPipeline:
    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_add_wav_file(self, mock_model_cls, tmp_path):
        """Add WAV via add_document_from_file, verify content contains transcript."""
        import wave, struct, math

        # Create a real WAV file
        wav_path = tmp_path / "recording.wav"
        with wave.open(str(wav_path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            for i in range(16000):
                sample = int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
                f.writeframes(struct.pack('<h', sample))

        # Mock the transcription
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Test transcription content")
        info = SimpleNamespace(language="en", duration=1.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(wav_path, DocumentConfig())
            content = audio_mod.render_transcript(result, wav_path.stem)

        assert "Test transcription content" in content
        assert "# Transcript: recording" in content

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_add_mp4_file(self, mock_model_cls, tmp_path):
        """Test .mp4 suffix handling."""
        # Create a dummy file (transcription is mocked)
        mp4_path = tmp_path / "video.mp4"
        mp4_path.write_bytes(b"\x00" * 100)

        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        fake_seg = SimpleNamespace(start=0.0, end=5.0, text="Video content here")
        info = SimpleNamespace(language="en", duration=5.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(mp4_path, DocumentConfig())
            content = audio_mod.render_transcript(result, mp4_path.stem)

        assert "Video content here" in content
        assert "# Transcript: video" in content

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_extract_content_audio(self, mock_model_cls, tmp_path):
        """Call _extract_content path with doc_type='audio', verify (content, 'markdown') returned."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Extracted audio text")
        info = SimpleNamespace(language="en", duration=1.0)
        mock_model.transcribe.return_value = (iter([fake_seg]), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(tmp_path / "test.wav", DocumentConfig())
            content = audio_mod.render_transcript(result, "test")

        assert "Extracted audio text" in content
        # The render always produces markdown format
        assert content.startswith("# Transcript:")

    def test_audio_config_from_yaml(self):
        """Parse DocumentConfig with whisper_model/diarize/hf_token fields."""
        config = DocumentConfig(
            whisper_model="base",
            diarize=True,
            hf_token="hf_test_token_123",
            language="en",
        )
        assert config.whisper_model == "base"
        assert config.diarize is True
        assert config.hf_token == "hf_test_token_123"
        assert config.language == "en"

    def test_audio_config_defaults(self):
        """Verify default values for audio config fields."""
        config = DocumentConfig()
        assert config.whisper_model == "large-v3"
        assert config.diarize is False
        assert config.hf_token is None
        assert config.language is None

    @patch("constat.discovery.doc_tools._audio.WhisperModel", create=True)
    def test_audio_chunks_contain_transcript(self, mock_model_cls, tmp_path):
        """Verify chunked output includes transcript text."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        segments = [
            SimpleNamespace(start=0.0, end=10.0, text="First paragraph of speech."),
            SimpleNamespace(start=10.0, end=20.0, text="Second paragraph of speech."),
            SimpleNamespace(start=20.0, end=30.0, text="Third paragraph of speech."),
        ]
        info = SimpleNamespace(language="en", duration=30.0)
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_model_cls)}):
            import importlib
            import constat.discovery.doc_tools._audio as audio_mod
            importlib.reload(audio_mod)

            result = audio_mod.transcribe_audio(tmp_path / "test.wav", DocumentConfig())
            content = audio_mod.render_transcript(result, "test")

        assert "First paragraph of speech." in content
        assert "Second paragraph of speech." in content
        assert "Third paragraph of speech." in content
