"""MinerU API client and output organizer.

Input: local PDF files and MinerU API responses.
Output: normalized pdf_parse/<doc_id> folders.
Side effects: calls MinerU, downloads zips, writes parsed files.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from backend.mineru.convert_middle_json import convert_middle_data_to_content_list
from backend.rag.doc_store import doc_dir_for_pdf, safe_doc_id, write_manifest

logger = logging.getLogger(__name__)

MINERU_BASE_URL = "https://mineru.net"


@dataclass(frozen=True)
class MinerUDownload:
    batch_id: str
    file_name: str
    zip_url: str
    zip_path: Path


@dataclass(frozen=True)
class MinerUArtifacts:
    content_list_path: Path
    markdown_path: Path | None
    middle_path: Path | None
    origin_pdf_path: Path
    manifest_path: Path
    extract_dir: Path


@dataclass(frozen=True)
class MinerUResult:
    file_name: str
    state: str
    full_zip_url: str = ""
    err_msg: str = ""
    data_id: str = ""
    progress: dict[str, Any] | None = None


ProgressCallback = Callable[[str], None]


class MinerUClient:
    def __init__(
        self,
        token: str,
        model_version: str = "vlm",
        language: str = "ch",
        poll_interval_seconds: int = 10,
        timeout_seconds: int = 3600,
        session: Any | None = None,
        base_url: str = MINERU_BASE_URL,
    ) -> None:
        if not token:
            raise ValueError("MINERU_API_TOKEN is not configured.")
        self.token = token
        self.model_version = model_version
        self.language = language
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests
        self.base_url = base_url.rstrip("/")

    def parse_file(
        self,
        pdf_path: Path,
        download_dir: Path,
        is_ocr: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> MinerUDownload:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        batch_id, upload_url = self.request_upload_url(pdf_path, is_ocr=is_ocr)
        _emit(on_progress, f"MinerU batch created: {batch_id}")
        self.upload_file(upload_url, pdf_path)
        _emit(on_progress, "PDF uploaded. Waiting for MinerU extraction...")

        result = self.wait_for_batch(batch_id, file_name=pdf_path.name, on_progress=on_progress)
        if not result.full_zip_url:
            raise RuntimeError("MinerU finished without full_zip_url.")

        batch_download_dir = download_dir / batch_id
        batch_download_dir.mkdir(parents=True, exist_ok=True)
        zip_path = batch_download_dir / f"{pdf_path.stem}.zip"
        self.download_file(result.full_zip_url, zip_path)
        _emit(on_progress, f"Downloaded MinerU result zip: {zip_path}")

        return MinerUDownload(
            batch_id=batch_id,
            file_name=result.file_name or pdf_path.name,
            zip_url=result.full_zip_url,
            zip_path=zip_path,
        )

    def request_upload_url(self, pdf_path: Path, is_ocr: bool = False) -> tuple[str, str]:
        url = f"{self.base_url}/api/v4/file-urls/batch"
        payload = {
            "files": [{"name": pdf_path.name, "data_id": _data_id(pdf_path)}],
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": True,
            "enable_formula": True,
            "is_ocr": is_ocr,
        }
        response = self.session.post(url, headers=self._json_headers(), json=payload)
        result = _response_json(response)
        data = result.get("data", {})
        batch_id = str(data.get("batch_id", ""))
        file_urls = data.get("file_urls", [])
        if not batch_id or not file_urls:
            raise RuntimeError(f"MinerU did not return upload URL: {result}")
        return batch_id, str(file_urls[0])

    def upload_file(self, upload_url: str, pdf_path: Path) -> None:
        with pdf_path.open("rb") as file:
            response = self.session.put(upload_url, data=file)
        if getattr(response, "status_code", 0) not in {200, 201, 204}:
            raise RuntimeError(f"MinerU file upload failed with status {response.status_code}.")

    def wait_for_batch(
        self,
        batch_id: str,
        file_name: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> MinerUResult:
        deadline = time.monotonic() + self.timeout_seconds
        last_message = ""
        while time.monotonic() < deadline:
            result = self.get_batch_result(batch_id, file_name=file_name)
            message = _result_message(result)
            if message != last_message:
                _emit(on_progress, message)
                last_message = message
            if result.state == "done":
                return result
            if result.state == "failed":
                raise RuntimeError(f"MinerU extraction failed: {result.err_msg or 'unknown error'}")
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"MinerU extraction timed out after {self.timeout_seconds} seconds.")

    def get_batch_result(self, batch_id: str, file_name: str | None = None) -> MinerUResult:
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        response = self.session.get(url, headers=self._json_headers())
        result = _response_json(response)
        data = result.get("data", {})
        raw_results = data.get("extract_result", [])
        if isinstance(raw_results, dict):
            raw_results = [raw_results]
        if not raw_results:
            raise RuntimeError(f"MinerU returned no extract_result for batch {batch_id}.")

        selected = raw_results[0]
        if file_name:
            selected = next(
                (item for item in raw_results if item.get("file_name") == file_name),
                selected,
            )
        return MinerUResult(
            file_name=str(selected.get("file_name", file_name or "")),
            state=str(selected.get("state", "")),
            full_zip_url=str(selected.get("full_zip_url", "")),
            err_msg=str(selected.get("err_msg", "")),
            data_id=str(selected.get("data_id", "")),
            progress=selected.get("extract_progress"),
        )

    def download_file(self, url: str, output_path: Path) -> None:
        response = self.session.get(url, stream=True)
        if getattr(response, "status_code", 0) != 200:
            raise RuntimeError(f"MinerU result download failed with status {response.status_code}.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            if hasattr(response, "iter_content"):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
            else:
                file.write(getattr(response, "content", b""))

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }


def organize_mineru_zip(zip_path: Path, pdf_path: Path, data_dir: Path) -> MinerUArtifacts:
    extract_dir = zip_path.parent / "extracted"
    _safe_extract_zip(zip_path, extract_dir)
    return organize_mineru_outputs(extract_dir, pdf_path, data_dir)


def organize_mineru_outputs(extract_dir: Path, pdf_path: Path, data_dir: Path) -> MinerUArtifacts:
    pdf_path = pdf_path.resolve()
    doc_id = safe_doc_id(pdf_path.stem)
    doc_dir = doc_dir_for_pdf(data_dir, pdf_path)
    doc_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = doc_dir / "_raw_mineru"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.copytree(extract_dir, raw_dir)

    _copy_resource_dirs(extract_dir, doc_dir)

    content_list_path = doc_dir / "content_list.json"
    markdown_path = _copy_first(
        _find_first(extract_dir, ["full.md", "*.md", "*.markdown"]),
        doc_dir / "full.md",
    )
    middle_path = _copy_first(
        _find_first(extract_dir, ["layout.json", "*_middle.json"]),
        doc_dir / "middle.json",
    )

    content_candidate = _find_content_list(extract_dir)
    if content_candidate is not None:
        shutil.copy2(content_candidate, content_list_path)
    else:
        middle_candidate = middle_path or _find_pdf_info_json(extract_dir)
        if middle_candidate is None:
            raise FileNotFoundError("No MinerU content_list JSON or pdf_info JSON found in extracted output.")
        _convert_fallback_content_list(middle_candidate, content_list_path)

    origin_pdf_path = doc_dir / "origin.pdf"
    shutil.copy2(pdf_path, origin_pdf_path)
    manifest_path = write_manifest(
        doc_dir,
        doc_id=doc_id,
        origin_pdf=origin_pdf_path,
        extra={
            "source_file": str(pdf_path),
            "mineru_raw_dir": "_raw_mineru",
        },
    )

    return MinerUArtifacts(
        content_list_path=content_list_path,
        markdown_path=markdown_path,
        middle_path=middle_path,
        origin_pdf_path=origin_pdf_path,
        manifest_path=manifest_path,
        extract_dir=extract_dir,
    )


def _response_json(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", 0)
    if status_code != 200:
        text = getattr(response, "text", "")
        raise RuntimeError(f"MinerU API request failed with status {status_code}: {text}")
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"MinerU API error: {result.get('msg', result)}")
    return result


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in MinerU zip: {member.filename}")
        archive.extractall(extract_dir)


def _copy_resource_dirs(source_dir: Path, target_dir: Path) -> None:
    for directory_name in ["images", "image", "assets"]:
        for source in sorted(path for path in source_dir.rglob(directory_name) if path.is_dir()):
            target = target_dir / directory_name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            return


def _find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _copy_first(source: Path | None, target: Path) -> Path | None:
    if source is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _find_content_list(root: Path) -> Path | None:
    candidates = sorted(
        path
        for path in root.rglob("*.json")
        if path.name == "content_list.json" or path.name.endswith("_content_list.json")
    )
    non_v2 = [path for path in candidates if "v2" not in path.stem]
    return (non_v2 or candidates or [None])[0]


def _find_pdf_info_json(root: Path) -> Path | None:
    preferred = _find_first(root, ["layout.json", "*_middle.json", "*.json"])
    candidates = [preferred] if preferred else []
    candidates.extend(path for path in sorted(root.rglob("*.json")) if path not in candidates)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("pdf_info"), list):
            return path
    return None


def _convert_fallback_content_list(source_json: Path, output_path: Path) -> None:
    with source_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    blocks = convert_middle_data_to_content_list(data)
    if not blocks:
        raise RuntimeError(f"No convertible MinerU blocks found in {source_json}.")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(blocks, file, ensure_ascii=False, indent=2)


def _data_id(path: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-")
    return (safe or "document")[:128]


def _result_message(result: MinerUResult) -> str:
    progress = result.progress or {}
    if progress:
        extracted = progress.get("extracted_pages", "?")
        total = progress.get("total_pages", "?")
        return f"MinerU state={result.state}, pages={extracted}/{total}"
    return f"MinerU state={result.state}"


def _emit(callback: ProgressCallback | None, message: str) -> None:
    logger.info(message)
    if callback is not None:
        callback(message)

