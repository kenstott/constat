# Copyright (c) 2025 Kenneth Stott
# Canary: 08a2c2f4-c8e0-40d4-a898-d89a477f355f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pure utility functions for document chunking — table/list detection, block merging, section extraction."""


def _is_table_line(line: str) -> bool:
    """Detect pipe-separated table rows (DOCX/XLSX/PPTX/HTML/MD formats)."""
    return line.count("|") >= 2


def _is_list_line(line: str) -> bool:
    """Detect markdown list items."""
    stripped = line.lstrip()
    if stripped[:2] in ("- ", "* ", "+ "):
        return True
    # Ordered list: "1. ", "2. ", etc.
    if stripped and stripped[0].isdigit():
        dot_pos = stripped.find(". ")
        if 0 < dot_pos <= 4:
            return stripped[:dot_pos].isdigit()
    return False


def _merge_blocks(paragraphs: list[str], separator: str) -> list[str]:
    """Merge consecutive table lines and list lines into atomic blocks."""
    merged: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        lines = para.split("\n") if "\n" in para else [para]

        # Check if this paragraph is a table block
        if any(_is_table_line(l) for l in lines if l.strip()):
            block_parts = [para]
            j = i + 1
            while j < len(paragraphs):
                next_lines = paragraphs[j].split("\n") if "\n" in paragraphs[j] else [paragraphs[j]]
                if any(_is_table_line(l) for l in next_lines if l.strip()):
                    block_parts.append(paragraphs[j])
                    j += 1
                else:
                    break
            merged.append(separator.join(block_parts))
            i = j
            continue

        # Check if this paragraph is a list block
        if any(_is_list_line(l) for l in lines if l.strip()):
            block_parts = [para]
            j = i + 1
            while j < len(paragraphs):
                next_lines = paragraphs[j].split("\n") if "\n" in paragraphs[j] else [paragraphs[j]]
                if any(_is_list_line(l) for l in next_lines if l.strip()):
                    block_parts.append(paragraphs[j])
                    j += 1
                else:
                    break
            merged.append(separator.join(block_parts))
            i = j
            continue

        merged.append(para)
        i += 1
    return merged


def _extract_markdown_sections(content: str, doc_format: str) -> list[str]:
    """Extract section headers from markdown content."""
    if doc_format not in ("markdown", "md"):
        return []
    return [line.lstrip("#").strip() for line in content.split("\n") if line.startswith("#")]


def chunk_document(
    name: str,
    content: str,
    chunk_size: int,
    table_chunk_limit: int,
) -> list:
    """Split a document into chunks for embedding.

    Chunks are split only on paragraph/line boundaries - never mid-paragraph.
    Paragraphs are combined until hitting chunk_size, then a new chunk starts.
    A chunk may exceed chunk_size if a single paragraph is larger (we don't split it).
    Table blocks and list blocks are merged into atomic units before chunking.

    Args:
        name: Document name for chunk metadata
        content: Full document text
        chunk_size: Target max characters per chunk
        table_chunk_limit: Max size for table blocks before row-level splitting

    Returns:
        List of DocumentChunk objects
    """
    from constat.discovery.models import DocumentChunk

    chunks = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)

    # Determine paragraph separator based on content
    # Use double newline if present, otherwise single newline
    if "\n\n" in content:
        paragraphs = content.split("\n\n")
        separator = "\n\n"
    else:
        paragraphs = content.split("\n")
        separator = "\n"

    # Build paragraph-to-offset map before merging
    para_offsets: list[int] = []
    pos = 0
    for para in paragraphs:
        idx = content.find(para, pos)
        para_offsets.append(len(content[:idx].encode("utf-8")) if idx >= 0 else -1)
        pos = idx + len(para) if idx >= 0 else pos

    # Merge consecutive table/list paragraphs into atomic blocks
    paragraphs = _merge_blocks(paragraphs, separator)

    chunk_index = 0
    current_chunk = ""
    current_section: str | None = None
    chunk_start_offset: int | None = None

    prev_heading_level: int | None = None
    # Track paragraph index across merged blocks
    raw_para_idx = 0

    def _make_chunk(text: str, section: str | None, idx: int, offset: int | None) -> DocumentChunk:
        src_len = len(text.encode("utf-8")) if text else 0
        return DocumentChunk(
            document_name=name,
            content=text,
            section=section,
            chunk_index=idx,
            source_offset=offset,
            source_length=src_len,
        )

    for para in paragraphs:
        para = para.strip()
        if not para:
            raw_para_idx += 1
            continue

        # Find this paragraph's byte offset in the original content
        para_byte_offset = content.find(para)
        if para_byte_offset >= 0:
            para_byte_offset = len(content[:para_byte_offset].encode("utf-8"))
        else:
            para_byte_offset = None

        # Detect heading level changes that should force a chunk break
        force_break = False
        prev_section = current_section

        # Track heading hierarchy as breadcrumbs
        if para.startswith("#"):
            level = len(para) - len(para.lstrip("#"))
            text = para.lstrip("#").strip()
            # Force chunk break on same-or-higher level heading (sibling/parent)
            if prev_heading_level is not None and level <= prev_heading_level and current_chunk:
                force_break = True
            prev_heading_level = level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            current_section = " > ".join(t for _, t in heading_stack)
        elif para.startswith("[Sheet:") and para.endswith("]"):
            if current_chunk:
                force_break = True
            sheet_name = para[len("[Sheet:"):-1].strip()
            heading_stack = [(0, sheet_name)]
            current_section = sheet_name
            prev_heading_level = 0

        if force_break:
            chunks.append(_make_chunk(current_chunk, prev_section, chunk_index, chunk_start_offset))
            chunk_index += 1
            current_chunk = ""
            chunk_start_offset = None

        # Check if adding this paragraph would exceed chunk size
        potential_chunk = (current_chunk + separator + para).strip() if current_chunk else para

        if len(potential_chunk) <= chunk_size:
            # Fits in current chunk - add it
            if not current_chunk:
                chunk_start_offset = para_byte_offset
            current_chunk = potential_chunk
        else:
            # Would exceed chunk size
            if current_chunk:
                # Save current chunk first
                chunks.append(_make_chunk(current_chunk, current_section, chunk_index, chunk_start_offset))
                chunk_index += 1
            # If oversized table block exceeds table_chunk_limit, split at row
            # boundaries with continuation markers for reassembly
            if len(para) > table_chunk_limit:
                lines = para.split("\n")
                is_table = any(_is_table_line(l) for l in lines if l.strip())
                sub_chunk = ""
                first_sub = True
                sub_start_offset = para_byte_offset
                for line in lines:
                    candidate = (sub_chunk + "\n" + line).strip() if sub_chunk else line
                    if len(candidate) <= chunk_size and sub_chunk:
                        sub_chunk = candidate
                    elif sub_chunk:
                        # Add table continuation markers
                        if is_table:
                            marker = "[TABLE:start]" if first_sub else "[TABLE:cont]"
                            sub_chunk = f"{marker}\n{sub_chunk}\n[TABLE:cont]"
                            first_sub = False
                        chunks.append(_make_chunk(sub_chunk, current_section, chunk_index, sub_start_offset))
                        chunk_index += 1
                        # Advance offset past emitted sub_chunk
                        if sub_start_offset is not None:
                            sub_start_offset += len(sub_chunk.encode("utf-8"))
                        sub_chunk = line
                    else:
                        sub_chunk = line
                # Mark last sub-chunk with [TABLE:end]
                if is_table and sub_chunk:
                    marker = "[TABLE:start]" if first_sub else "[TABLE:cont]"
                    sub_chunk = f"{marker}\n{sub_chunk}\n[TABLE:end]"
                current_chunk = sub_chunk
                chunk_start_offset = sub_start_offset
            else:
                # Start new chunk with this paragraph (even if it exceeds chunk_size)
                current_chunk = para
                chunk_start_offset = para_byte_offset

        raw_para_idx += 1

    # Save final chunk
    if current_chunk:
        chunks.append(_make_chunk(current_chunk, current_section, chunk_index, chunk_start_offset))

    return chunks
