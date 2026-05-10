# Copyright (c) 2025 Kenneth Stott
# Canary: 08a2c2f4-c8e0-40d4-a898-d89a477f355f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document loading dispatch logic — IMAP, file/directory, crawler, HTTP fetch."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from constat.core.config import DocumentConfig
from constat.discovery.models import LoadedDocument
from ._chunking import _extract_markdown_sections
from ._file_extractors import (
    _extract_pdf_text_from_bytes,
    _extract_docx_text_from_bytes,
    _extract_xlsx_text_from_bytes,
    _extract_pptx_text_from_bytes,
    _convert_html_to_markdown,
)
from ._mime import normalize_type, detect_type_from_source, is_binary_type
from ._transport import fetch_document, infer_transport, FetchResult
from ._schema_inference import _infer_structured_schema

logger = logging.getLogger(__name__)


def _extract_content(mixin, result: FetchResult, doc_type: str, name: str = "") -> tuple[str, str]:
    """Extract text content from FetchResult bytes based on doc_type.

    Args:
        mixin: The _CoreMixin instance (needed for _router, _resolve_doc_config, config)
        result: FetchResult with raw bytes
        doc_type: Detected document type
        name: Document name (for audio config lookup)

    Returns:
        (content, doc_format) tuple.
    """
    if doc_type == "pdf":
        return _extract_pdf_text_from_bytes(result.data), "text"
    elif doc_type == "docx":
        return _extract_docx_text_from_bytes(result.data), "text"
    elif doc_type == "xlsx":
        return _extract_xlsx_text_from_bytes(result.data), "text"
    elif doc_type == "pptx":
        return _extract_pptx_text_from_bytes(result.data), "text"
    elif doc_type == "image":
        from ._image import _extract_image, _render_image_result, _describe_image_sync, _ocr_via_vision
        image_result = _extract_image(
            path=Path(result.source_path) if result.source_path else None,
            data=result.data,
        )
        mime = result.detected_mime or "image/png"
        # Fallback to LLM vision OCR if tesseract failed
        if not image_result.ocr_text and mixin._router:
            logger.warning("Image %s: Tesseract OCR returned no text, falling back to LLM vision OCR",
                           result.source_path or "<bytes>")
            vision_ocr = _ocr_via_vision(mixin._router, result.data, mime)
            if vision_ocr.text:
                image_result.ocr_text = vision_ocr.text
                image_result.ocr_confidence = vision_ocr.mean_confidence
                image_result.ocr_word_count = vision_ocr.word_count
                from ._image import _classify_image
                image_result.category = _classify_image(vision_ocr)
        if mixin._router and image_result.category == "image-primary":
            try:
                desc = _describe_image_sync(mixin._router, result.data, mime)
                image_result.description = desc.get("description")
                image_result.subcategory = desc.get("subcategory", image_result.subcategory)
                image_result.labels = desc.get("labels", image_result.labels)
            except Exception:
                pass  # fall back to OCR-only
        img_name = Path(result.source_path).stem if result.source_path else "image"
        return _render_image_result(image_result, img_name), "markdown"
    elif doc_type == "audio":
        from ._audio import transcribe_audio, render_transcript as _render_transcript
        import tempfile
        suffix = Path(result.source_path).suffix if result.source_path else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(result.data)
            temp_path = Path(f.name)
        try:
            doc_config = mixin._resolve_doc_config(name) if name in mixin.config.documents else DocumentConfig()
            tr = transcribe_audio(temp_path, doc_config)
            name_stem = Path(result.source_path).stem if result.source_path else "audio"
            return _render_transcript(tr, name_stem), "markdown"
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        return result.data.decode("utf-8"), doc_type


def _load_documents_parallel(mixin, doc_names: list[str]) -> None:
    """Load multiple documents in parallel using threads (I/O bound)."""
    if len(doc_names) <= 1:
        for name in doc_names:
            try:
                _load_document(mixin, name)
            except Exception as e:
                logger.warning(f"[DOC_INIT] Failed to load {name}: {e}")
        return

    max_workers = min(len(doc_names), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_document, mixin, name): name for name in doc_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.warning(f"[DOC_INIT] Failed to load {name}: {e}")


def _load_document(mixin, name: str) -> dict | None:
    """Load a document from its configured source.

    Args:
        mixin: The _CoreMixin instance
        name: Document name

    Returns:
        None for text documents (stored in mixin._loaded_documents)
        dict for IMAP (with chunk counts)
    """
    from datetime import datetime
    from ._crawler import crawl_document as _crawl_document

    doc_config = mixin._resolve_doc_config(name)
    user_type = normalize_type(doc_config.type)
    transport = infer_transport(doc_config)

    # IMAP: fetch messages, chunk/embed each incrementally
    if transport == "imap":
        return _load_imap(mixin, name, doc_config, user_type)

    # SharePoint: discover site, fetch libraries/lists/calendars/pages
    if transport == "sharepoint":
        return _load_sharepoint(mixin, name, doc_config)

    # Cloud drive: list + download files from Google Drive / OneDrive
    if transport == "drive":
        return _load_drive(mixin, name, doc_config)

    # Directory: iterate files and load each as a sub-document
    if transport == "file":
        dir_path = Path(doc_config.path)
        if not dir_path.is_absolute() and mixin.config.config_dir:
            dir_path = (Path(mixin.config.config_dir) / doc_config.path).resolve()
        if dir_path.is_dir():
            loaded = 0
            for file_path in sorted(dir_path.rglob("*")):
                if not file_path.is_file():
                    continue
                sub_name = f"{name}:{file_path.name}"
                sub_result = FetchResult(
                    data=file_path.read_bytes(),
                    detected_mime=None,
                    source_path=str(file_path),
                )
                sub_type = detect_type_from_source(str(file_path), None)
                try:
                    sub_content, sub_format = _extract_content(mixin, sub_result, sub_type, name)
                except Exception as e:
                    logger.debug(f"Skipping {file_path}: {e}")
                    continue
                if sub_format == "html":
                    sub_content = _convert_html_to_markdown(sub_content)
                    sub_format = "markdown"
                mixin._loaded_documents[sub_name] = LoadedDocument(
                    name=sub_name,
                    config=doc_config,
                    content=sub_content,
                    format=sub_format,
                    sections=_extract_markdown_sections(sub_content, sub_format),
                    loaded_at=datetime.now().isoformat(),
                )
                loaded += 1
            logger.info(f"[DOC_INIT] Loaded {loaded} files from directory {dir_path}")
            return None

    # Fetch via transport (or crawl if follow_links)
    if doc_config.follow_links and doc_config.url:
        results = _crawl_document(doc_config, mixin.config.config_dir, fetch_document)
        # Use root document as primary
        if not results:
            raise ValueError(f"Crawler returned no results for {name}")
        _, root_result = results[0]
        result = root_result

        # Extract content from linked docs in parallel (CPU-bound: HTML→markdown, text extraction)
        def _extract_linked(i_url_result):
            i, url, linked_result = i_url_result
            linked_type = user_type if user_type != "auto" else detect_type_from_source(
                linked_result.source_path, linked_result.detected_mime
            )
            if is_binary_type(linked_type):
                logger.debug(f"Skipping binary linked doc: {url}")
                return None
            try:
                linked_content, linked_format = _extract_content(mixin, linked_result, linked_type, name)
            except (UnicodeDecodeError, ValueError) as e:
                logger.debug(f"Skipping non-decodable linked doc {url}: {e}")
                return None
            if linked_format == "html":
                linked_content = _convert_html_to_markdown(linked_content)
                linked_format = "markdown"
            sub_name = f"{name}:crawled_{i}"
            return LoadedDocument(
                name=sub_name, config=doc_config, content=linked_content,
                format=linked_format,
                sections=_extract_markdown_sections(linked_content, linked_format),
                loaded_at=datetime.now().isoformat(), source_url=url,
            )

        linked_items = [(i, url, lr) for i, (url, lr) in enumerate(results[1:], 1)]
        logger.info(f"[DOC_INIT] Extracting content from {len(linked_items)} crawled pages")
        if len(linked_items) <= 2:
            for item in linked_items:
                doc = _extract_linked(item)
                if doc:
                    mixin._loaded_documents[doc.name] = doc
        else:
            max_workers = min(len(linked_items), 8)
            logger.info(f"[DOC_INIT] Parallel extraction with {max_workers} threads")
            extracted = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for doc in pool.map(_extract_linked, linked_items):
                    if doc:
                        mixin._loaded_documents[doc.name] = doc
                        extracted += 1
            logger.info(f"[DOC_INIT] Extracted {extracted}/{len(linked_items)} crawled pages")
    else:
        result = fetch_document(doc_config, mixin.config.config_dir)

    # Resolve type
    doc_type = user_type if user_type != "auto" else detect_type_from_source(
        result.source_path, result.detected_mime
    )

    # Structured schema check for local files
    if transport == "file" and not is_binary_type(doc_type) and result.source_path:
        schema = _infer_structured_schema(Path(result.source_path), doc_config.description)
        if schema:
            content = schema.to_metadata_doc()
            doc_format = schema.file_format
            mixin._loaded_documents[name] = LoadedDocument(
                name=name,
                config=doc_config,
                content=content,
                format=doc_format,
                sections=["Schema", "Columns"],
                loaded_at=datetime.now().isoformat(),
            )
            return None

    # Extract content from bytes
    content, doc_format = _extract_content(mixin, result, doc_type, name)

    # HTML -> markdown conversion
    if doc_format == "html":
        content = _convert_html_to_markdown(content)
        doc_format = "markdown"

    mixin._loaded_documents[name] = LoadedDocument(
        name=name,
        config=doc_config,
        content=content,
        format=doc_format,
        sections=_extract_markdown_sections(content, doc_format),
        loaded_at=datetime.now().isoformat(),
    )

    # Extract embedded images if enabled
    if doc_config.extract_images and doc_type in ("pdf", "docx", "pptx", "xlsx"):
        from ._file_extractors import _extract_images_from_document
        from ._image import _extract_image, _render_image_result, _describe_image_sync

        embedded_images = _extract_images_from_document(
            path=Path(result.source_path) if result.source_path else None,
            data=result.data,
            doc_type=doc_type,
            config_dir=mixin.config.config_dir,
        )
        logger.info("Document %s: extracted %d embedded images", name, len(embedded_images))
        vision_calls = 0
        for img in embedded_images:
            img_name = f"{name}:{img.name}"
            image_result = _extract_image(path=None, data=img.data)
            logger.info("Embedded image %s: category=%s, ocr_words=%d, ocr_text=%r",
                         img_name, image_result.category, image_result.ocr_word_count,
                         image_result.ocr_text[:100] if image_result.ocr_text else "")
            if image_result.category == "image-primary" and mixin._router and vision_calls < 50:
                try:
                    desc = _describe_image_sync(mixin._router, img.data, img.mime_type)
                    image_result.description = desc.get("description")
                    image_result.subcategory = desc.get("subcategory", image_result.subcategory)
                    image_result.labels = desc.get("labels", image_result.labels)
                    vision_calls += 1
                    logger.info("Embedded image %s: vision description=%r, labels=%s",
                                 img_name, image_result.description or "", image_result.labels)
                except Exception as e:
                    logger.warning("Embedded image %s: vision description failed: %s", img_name, e)
            if image_result.labels:
                mixin._image_labels.extend(image_result.labels)
            img_content = _render_image_result(image_result, img.name)
            mixin._loaded_documents[img_name] = LoadedDocument(
                name=img_name,
                config=doc_config,
                content=img_content,
                format="markdown",
                sections=[],
                loaded_at=datetime.now().isoformat(),
            )


def _load_imap(mixin, name: str, doc_config: DocumentConfig, user_type: str) -> dict:
    """Handle IMAP document loading — fetch messages, chunk/embed each incrementally."""
    from datetime import datetime
    from ._imap import IMAPFetcher, _render_email

    logger.info("[IMAP] Connecting to %s mailbox=%s (auth=%s, since=%s)",
                doc_config.url, doc_config.mailbox, doc_config.auth_type, doc_config.since)
    fetcher = IMAPFetcher(doc_config, config_dir=Path(mixin.config.config_dir) if mixin.config.config_dir else None)
    messages = fetcher.fetch_messages()
    logger.info("[IMAP] Fetched %d messages from %s/%s", len(messages), name, doc_config.mailbox)

    imap_ctx = getattr(mixin, '_imap_context', None)

    total_attachments = 0
    total_chunks = 0
    for i, msg in enumerate(messages):
        msg_name = f"{name}:{msg.message_id}"

        # Email body -> LoadedDocument
        body_text = _render_email(msg, doc_config.include_headers)
        mixin._loaded_documents[msg_name] = LoadedDocument(
            name=msg_name,
            config=doc_config,
            content=body_text,
            format="markdown",
            sections=_extract_markdown_sections(body_text, "markdown"),
            loaded_at=datetime.now().isoformat(),
        )

        # Chunk/embed this message immediately
        if imap_ctx:
            total_chunks += mixin._index_loaded_doc(
                msg_name, imap_ctx["domain_id"], imap_ctx["session_id"],
                imap_ctx["skip_entity_extraction"],
            )

        # Attachments
        if doc_config.extract_attachments:
            att_chunks = _load_imap_attachments(
                mixin, msg, msg_name, doc_config, imap_ctx,
            )
            total_attachments += att_chunks["attachment_count"]
            total_chunks += att_chunks["chunk_count"]

        if (i + 1) % 50 == 0:
            logger.info("[IMAP] Progress: %d/%d messages, %d chunks",
                        i + 1, len(messages), total_chunks)
        if imap_ctx and imap_ctx.get("progress_callback"):
            imap_ctx["progress_callback"](name, i + 1, len(messages))

    logger.info("[IMAP] Done: %s — %d messages, %d attachments, %d chunks",
                name, len(messages), total_attachments, total_chunks)
    return {"imap_chunks": total_chunks, "imap_docs": len(messages)}


def _load_imap_attachments(mixin, msg, msg_name, doc_config, imap_ctx) -> dict:
    """Process attachments for a single IMAP message. Returns counts."""
    from datetime import datetime

    attachment_count = 0
    chunk_count = 0

    for att in msg.attachments:
        att_name = f"{msg_name}:{att.filename}"
        att_type = detect_type_from_source(att.filename, att.content_type)
        attachment_count += 1
        try:
            if att_type == "image":
                from ._image import _extract_image, _render_image_result, _describe_image_sync
                img_result = _extract_image(data=att.data)
                if img_result.category == "image-primary" and mixin._router:
                    desc = _describe_image_sync(mixin._router, att.data, att.content_type)
                    img_result.description = desc.get("description")
                    img_result.subcategory = desc.get("subcategory", "other")
                    img_result.labels = desc.get("labels", [])
                content = _render_image_result(img_result, att.filename)
                fmt = "markdown"
            else:
                fetch_result = FetchResult(data=att.data, detected_mime=att.content_type, source_path=att.filename)
                content, fmt = _extract_content(mixin, fetch_result, att_type, "")
        except Exception as att_err:
            logger.warning("[IMAP]   Skipping attachment %s: %s", att.filename, att_err)
            continue

        # Extract embedded images from PDF/Office attachments
        if doc_config.extract_images and att_type in ("pdf", "docx", "pptx", "xlsx"):
            content, fmt, extra_chunks = _extract_embedded_images_from_attachment(
                mixin, att_name, att, att_type, doc_config, imap_ctx, content, fmt,
            )
            chunk_count += extra_chunks

        # After image extraction, if content is still empty, store placeholder
        if not content.strip() and att_type in ("pdf", "docx", "xlsx", "pptx"):
            content = f"[{att_type.upper()} attachment: {att.filename} — no extractable text]"
            fmt = "text"
        mixin._loaded_documents[att_name] = LoadedDocument(
            name=att_name, config=doc_config, content=content, format=fmt,
            sections=[], loaded_at=datetime.now().isoformat(),
        )
        if imap_ctx:
            chunk_count += mixin._index_loaded_doc(
                att_name, imap_ctx["domain_id"], imap_ctx["session_id"],
                imap_ctx["skip_entity_extraction"],
            )

    return {"attachment_count": attachment_count, "chunk_count": chunk_count}


def _extract_embedded_images_from_attachment(
    mixin, att_name, att, att_type, doc_config, imap_ctx, content, fmt,
) -> tuple[str, str, int]:
    """Extract embedded images from a PDF/Office attachment. Returns (content, fmt, chunk_count)."""
    from datetime import datetime
    from ._file_extractors import _extract_images_from_document
    from ._image import _extract_image, _render_image_result, _describe_image_sync

    chunk_count = 0
    try:
        embedded_images = _extract_images_from_document(
            path=None, data=att.data, doc_type=att_type,
        )
        ocr_parts = []
        for img in embedded_images:
            img_doc_name = f"{att_name}:{img.name}"
            image_result = _extract_image(path=None, data=img.data)
            if image_result.category == "image-primary" and mixin._router:
                try:
                    desc = _describe_image_sync(mixin._router, img.data, img.mime_type)
                    image_result.description = desc.get("description")
                    image_result.subcategory = desc.get("subcategory", image_result.subcategory)
                    image_result.labels = desc.get("labels", image_result.labels)
                except Exception:
                    pass
            if image_result.labels:
                mixin._image_labels.extend(image_result.labels)
            img_content = _render_image_result(image_result, img.name)
            mixin._loaded_documents[img_doc_name] = LoadedDocument(
                name=img_doc_name, config=doc_config, content=img_content,
                format="markdown", sections=[], loaded_at=datetime.now().isoformat(),
            )
            if imap_ctx:
                chunk_count += mixin._index_loaded_doc(
                    img_doc_name, imap_ctx["domain_id"], imap_ctx["session_id"],
                    imap_ctx["skip_entity_extraction"],
                )
            # Collect OCR text to enrich empty parent documents
            if image_result.ocr_text and image_result.ocr_text.strip():
                page_label = f"[Page {img.page}]" if img.page else f"[Image {img.index}]"
                ocr_parts.append(f"{page_label}\n{image_result.ocr_text.strip()}")
        if embedded_images:
            logger.info("[IMAP]   %s: extracted %d embedded images", att.filename, len(embedded_images))
        # If parent text extraction was empty, use OCR text from images
        if not content.strip() and ocr_parts:
            content = "\n\n".join(ocr_parts)
            fmt = "text"
            logger.info("[IMAP]   %s: using OCR text from %d embedded images", att.filename, len(ocr_parts))
    except Exception as img_err:
        logger.warning("[IMAP]   Embedded image extraction failed for %s: %s", att.filename, img_err)

    return content, fmt, chunk_count


def _load_file_directly(mixin, name: str, filepath: Path, doc_config: DocumentConfig) -> None:
    """Load a file directly from a path (for expanded glob/directory entries)."""
    from datetime import datetime

    # Check if it's a structured data file - use schema metadata instead of raw content
    schema = _infer_structured_schema(filepath, doc_config.description)
    if schema:
        content = schema.to_metadata_doc()
        doc_format = schema.file_format

        file_config = DocumentConfig(
            path=str(filepath),
            description=doc_config.description,
            tags=doc_config.tags,
        )

        mixin._loaded_documents[name] = LoadedDocument(
            name=name,
            config=file_config,
            content=content,
            format=doc_format,
            sections=["Schema", "Columns"],
            loaded_at=datetime.now().isoformat(),
        )
        return

    # Use transport to read file bytes
    file_result = FetchResult(
        data=filepath.read_bytes(),
        detected_mime=None,
        source_path=str(filepath),
    )

    # Resolve type from user config or auto-detect
    user_type = normalize_type(doc_config.type)
    doc_type = user_type if user_type != "auto" else detect_type_from_source(
        str(filepath), None
    )

    # Extract content
    content, doc_format = _extract_content(mixin, file_result, doc_type, name)

    # Convert HTML to markdown
    if doc_format == "html":
        content = _convert_html_to_markdown(content)
        doc_format = "markdown"

    file_config = DocumentConfig(
        path=str(filepath),
        description=doc_config.description,
        tags=doc_config.tags,
    )

    mixin._loaded_documents[name] = LoadedDocument(
        name=name,
        config=file_config,
        content=content,
        format=doc_format,
        sections=_extract_markdown_sections(content, doc_format),
        loaded_at=datetime.now().isoformat(),
    )


def _load_drive(mixin, name: str, doc_config: DocumentConfig) -> None:
    """Handle cloud drive document loading — list files, download, extract content."""
    from datetime import datetime
    from ._drive import DriveFetcher

    logger.info("[DRIVE] Listing files from %s provider=%s", name, doc_config.provider)
    fetcher = DriveFetcher(
        doc_config,
        config_dir=Path(mixin.config.config_dir) if mixin.config.config_dir else None,
    )
    files = fetcher.list_files()
    logger.info("[DRIVE] Found %d files in %s", len(files), name)

    # Google native format → exported Office doc type
    _native_export_types = {
        "application/vnd.google-apps.document": "docx",
        "application/vnd.google-apps.spreadsheet": "xlsx",
        "application/vnd.google-apps.presentation": "pptx",
    }

    loaded = 0
    for file in files:
        file_id = fetcher.make_file_id(file)
        file_name = f"{name}:{file_id}"

        data = fetcher.download_file(file)

        # Determine doc type
        if file.is_google_native:
            doc_type = _native_export_types.get(file.mime_type, "auto")
        else:
            doc_type = detect_type_from_source(file.name, file.mime_type)

        sub_result = FetchResult(
            data=data,
            detected_mime=file.mime_type,
            source_path=file.name,
        )

        content, doc_format = _extract_content(mixin, sub_result, doc_type, name)

        if doc_format == "html":
            content = _convert_html_to_markdown(content)
            doc_format = "markdown"

        mixin._loaded_documents[file_name] = LoadedDocument(
            name=file_name,
            config=doc_config,
            content=content,
            format=doc_format,
            sections=_extract_markdown_sections(content, doc_format),
            loaded_at=datetime.now().isoformat(),
        )
        loaded += 1

    logger.info("[DRIVE] Loaded %d files from %s", loaded, name)
    return None


def _load_sharepoint(mixin, name: str, doc_config: DocumentConfig) -> None:
    """Handle SharePoint document loading — discover site, fetch all resource types."""
    from datetime import datetime
    from ._sharepoint import SharePointClient
    from ._calendar import render_event
    from ._drive import DriveFetcher

    logger.info("[SHAREPOINT] Discovering site %s", doc_config.site_url)
    client = SharePointClient(
        doc_config,
        config_dir=Path(mixin.config.config_dir) if mixin.config.config_dir else None,
    )
    site = client.discover_site()
    logger.info(
        "[SHAREPOINT] Found %d libraries, %d lists, %d calendars, %d pages",
        len(site["libraries"]),
        len(site["lists"]),
        len(site["calendars"]),
        len(site["pages"]),
    )

    loaded = 0

    # Libraries: list files, download, extract (reuse drive patterns)
    for lib in site["libraries"]:
        files = client.fetch_library_files(lib)
        for file in files:
            file_id = DriveFetcher.make_file_id(file)
            file_name = f"{name}:lib:{file_id}"

            data = client.download_library_file(lib, file)
            doc_type = detect_type_from_source(file.name, file.mime_type)
            sub_result = FetchResult(
                data=data,
                detected_mime=file.mime_type,
                source_path=file.name,
            )
            content, doc_format = _extract_content(mixin, sub_result, doc_type, name)
            if doc_format == "html":
                content = _convert_html_to_markdown(content)
                doc_format = "markdown"
            mixin._loaded_documents[file_name] = LoadedDocument(
                name=file_name,
                config=doc_config,
                content=content,
                format=doc_format,
                sections=_extract_markdown_sections(content, doc_format),
                loaded_at=datetime.now().isoformat(),
            )
            loaded += 1

    # Lists: fetch items, render as markdown
    for sp_list in site["lists"]:
        items = client.fetch_list_items(sp_list)
        list_doc_name = f"{name}:list:{sp_list.name}"
        content = client.render_list_as_markdown(sp_list, items)
        mixin._loaded_documents[list_doc_name] = LoadedDocument(
            name=list_doc_name,
            config=doc_config,
            content=content,
            format="markdown",
            sections=_extract_markdown_sections(content, "markdown"),
            loaded_at=datetime.now().isoformat(),
        )
        loaded += 1

    # Calendars: fetch events, render
    for cal in site["calendars"]:
        events = client.fetch_calendar_events(cal)
        for event in events:
            event_name = f"{name}:cal:{event.event_id}"
            content = render_event(event)
            mixin._loaded_documents[event_name] = LoadedDocument(
                name=event_name,
                config=doc_config,
                content=content,
                format="markdown",
                sections=_extract_markdown_sections(content, "markdown"),
                loaded_at=datetime.now().isoformat(),
            )
            loaded += 1

    # Pages: store as markdown documents
    for page in site["pages"]:
        page_name = f"{name}:page:{page.id}"
        mixin._loaded_documents[page_name] = LoadedDocument(
            name=page_name,
            config=doc_config,
            content=page.content,
            format="markdown",
            sections=_extract_markdown_sections(page.content, "markdown"),
            loaded_at=datetime.now().isoformat(),
        )
        loaded += 1

    logger.info("[SHAREPOINT] Loaded %d resources from %s", loaded, name)
    return None


def add_document_from_config(
    mixin,
    name: str,
    doc_config: DocumentConfig,
    domain_id: str | None = None,
    session_id: str | None = None,
    skip_entity_extraction: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[bool, str]:
    """Add a document from a DocumentConfig (supports URL, inline, etc.).

    Temporarily registers the config so _resolve_doc_config can find it,
    loads via _load_document (handles URL fetch, crawl, HTML->markdown),
    then indexes all loaded docs with domain/session scoping.

    Args:
        mixin: The _CoreMixin instance
        name: Document name
        doc_config: DocumentConfig with url, content, etc.
        domain_id: Domain this document belongs to
        session_id: Session this document was added in
        skip_entity_extraction: If True, skip NER
        progress_callback: Optional callback for IMAP progress

    Returns:
        Tuple of (success, message)
    """
    if mixin.config.documents is None:
        mixin.config.documents = {}

    # Temporarily register so _resolve_doc_config works
    mixin.config.documents[name] = doc_config

    try:
        # For IMAP, set context so _load_document can chunk/embed per message
        mixin._imap_context = {
            "domain_id": domain_id, "session_id": session_id,
            "skip_entity_extraction": skip_entity_extraction,
            "progress_callback": progress_callback,
        }

        result = _load_document(mixin, name)

        mixin._imap_context = None

        # IMAP handles chunking inline per message — already done
        if isinstance(result, dict) and "imap_chunks" in result:
            return True, f"Added {result['imap_docs']} email(s) ({result['imap_chunks']} chunks)"

        # Non-IMAP: collect loaded docs and chunk/embed in batch
        loaded_names = [
            n for n in mixin._loaded_documents
            if n == name or n.startswith(f"{name}:")
        ]

        logger.info(f"[DOC_ADD] Indexing {len(loaded_names)} documents for '{name}'")

        if len(loaded_names) > 1:
            total_chunks = mixin._index_loaded_docs_batch(
                loaded_names, domain_id, session_id, skip_entity_extraction,
            )
        else:
            total_chunks = 0
            for doc_name in loaded_names:
                total_chunks += mixin._index_loaded_doc(
                    doc_name, domain_id, session_id, skip_entity_extraction,
                )

        for doc_name in loaded_names:
            doc = mixin._loaded_documents.get(doc_name)
            if doc and getattr(doc, 'source_url', None) and hasattr(mixin._vector_store, 'store_document_url'):
                mixin._vector_store.store_document_url(doc_name, doc.source_url)

        logger.info(f"[DOC_ADD] Indexed {total_chunks} chunks from {len(loaded_names)} documents")
        return True, f"Added {len(loaded_names)} document(s) from '{name}' ({total_chunks} chunks)"
    except Exception as e:
        return False, f"Failed to load document from config: {e}"
    finally:
        mixin._imap_context = None
        mixin.config.documents.pop(name, None)
