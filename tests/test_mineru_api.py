from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from backend.mineru.api import MinerUClient, organize_mineru_zip


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.content = content
        self.text = text

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_content(self, chunk_size: int = 1024):
        yield self.content


class FakeSession:
    def __init__(self, zip_bytes: bytes, states: list[dict[str, Any]]) -> None:
        self.zip_bytes = zip_bytes
        self.states = states
        self.put_kwargs: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url.endswith("/api/v4/file-urls/batch")
        return FakeResponse(
            payload={
                "code": 0,
                "msg": "ok",
                "data": {
                    "batch_id": "batch-1",
                    "file_urls": ["https://upload.example/demo.pdf"],
                },
            }
        )

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url == "https://upload.example/demo.pdf"
        self.put_kwargs = kwargs
        return FakeResponse(status_code=200)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/v4/extract-results/batch/batch-1"):
            state = self.states.pop(0)
            return FakeResponse(
                payload={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "extract_result": [state],
                    },
                }
            )
        if url == "https://cdn.example/result.zip":
            return FakeResponse(content=self.zip_bytes)
        raise AssertionError(f"unexpected GET url: {url}")


def test_mineru_client_uploads_polls_and_downloads_zip(tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    zip_bytes = _zip_bytes({"demo_content_list.json": json.dumps([{"type": "text", "text": "ok"}])})
    session = FakeSession(
        zip_bytes=zip_bytes,
        states=[
            {"file_name": "demo.pdf", "state": "waiting-file"},
            {"file_name": "demo.pdf", "state": "running", "extract_progress": {"extracted_pages": 1, "total_pages": 2}},
            {"file_name": "demo.pdf", "state": "converting"},
            {"file_name": "demo.pdf", "state": "done", "full_zip_url": "https://cdn.example/result.zip"},
        ],
    )
    messages: list[str] = []

    client = MinerUClient(
        token="token",
        poll_interval_seconds=0,
        timeout_seconds=10,
        session=session,
    )
    download = client.parse_file(pdf_path, tmp_path / "downloads", on_progress=messages.append)

    assert download.batch_id == "batch-1"
    assert download.zip_path.read_bytes() == zip_bytes
    assert session.put_kwargs is not None
    assert "headers" not in session.put_kwargs
    assert any("state=running" in message for message in messages)


def test_mineru_client_raises_on_failed_state(tmp_path: Path) -> None:
    session = FakeSession(
        zip_bytes=b"",
        states=[{"file_name": "demo.pdf", "state": "failed", "err_msg": "bad pdf"}],
    )
    client = MinerUClient(
        token="token",
        poll_interval_seconds=0,
        timeout_seconds=10,
        session=session,
    )

    with pytest.raises(RuntimeError, match="bad pdf"):
        client.wait_for_batch("batch-1", file_name="demo.pdf")


def test_mineru_client_times_out() -> None:
    session = FakeSession(
        zip_bytes=b"",
        states=[{"file_name": "demo.pdf", "state": "running"}],
    )
    client = MinerUClient(
        token="token",
        poll_interval_seconds=0,
        timeout_seconds=0,
        session=session,
    )

    with pytest.raises(TimeoutError):
        client.wait_for_batch("batch-1", file_name="demo.pdf")


def test_organize_mineru_zip_copies_outputs_and_origin_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    zip_path = tmp_path / "result.zip"
    zip_path.write_bytes(
        _zip_bytes(
            {
                "full.md": "# Manual\n",
                "layout.json": json.dumps({"pdf_info": []}),
                "manual_content_list.json": json.dumps([{"type": "text", "text": "Manual", "page_idx": 0}]),
                "images/a.jpg": "image",
            }
        )
    )

    artifacts = organize_mineru_zip(zip_path, pdf_path, tmp_path / "data")

    assert artifacts.content_list_path == tmp_path / "data" / "manual" / "content_list.json"
    assert artifacts.content_list_path.exists()
    assert artifacts.markdown_path is not None
    assert artifacts.markdown_path.read_text(encoding="utf-8") == "# Manual\n"
    assert artifacts.middle_path is not None
    assert artifacts.origin_pdf_path.read_bytes() == b"%PDF-1.4"
    assert artifacts.manifest_path.exists()
    assert (tmp_path / "data" / "manual" / "images" / "a.jpg").exists()


def test_organize_mineru_zip_converts_layout_when_content_list_missing(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    zip_path = tmp_path / "result.zip"
    layout = {
        "pdf_info": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "type": "title",
                        "bbox": [0, 0, 100, 50],
                        "lines": [{"spans": [{"content": "Manual Title"}]}],
                    }
                ],
            }
        ]
    }
    zip_path.write_bytes(_zip_bytes({"layout.json": json.dumps(layout)}))

    artifacts = organize_mineru_zip(zip_path, pdf_path, tmp_path / "data")
    content = json.loads(artifacts.content_list_path.read_text(encoding="utf-8"))

    assert content == [
        {
            "type": "text",
            "text": "Manual Title",
            "bbox": [0, 0, 100, 50],
            "page_idx": 0,
            "text_level": 1,
        }
    ]


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()

