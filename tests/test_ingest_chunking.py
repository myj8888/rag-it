from pathlib import Path

from backend.rag.chunking import build_chunks
from backend.rag.ingest import html_to_text, load_mineru_blocks


def test_html_to_text_extracts_table_cells() -> None:
    html = "<table><tr><td>场景</td><td>方案</td></tr><tr><td>RAG</td><td>Qdrant</td></tr></table>"
    assert "场景 | 方案" in html_to_text(html)
    assert "RAG | Qdrant" in html_to_text(html)


def test_load_mineru_blocks_and_build_chunks(tmp_path: Path) -> None:
    data_dir = _sample_mineru_dir(tmp_path)
    blocks = load_mineru_blocks(data_dir, enable_image_ocr=False, enable_pdf_links=False)
    chunks = build_chunks(blocks)
    assert blocks
    assert chunks
    assert any("MinerU" in chunk.text for chunk in chunks)
    assert any(chunk.block_type == "table" for chunk in chunks)


def test_build_chunks_attaches_same_page_related_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "manual"
    data_dir.mkdir()
    (data_dir / "content_list.json").write_text(
        """
[
  {"type": "image", "img_path": "images/step.jpg", "bbox": [0, 100, 300, 300], "page_idx": 0},
  {"type": "text", "text": "按图示方向更换墨盒。", "bbox": [350, 120, 900, 180], "page_idx": 0}
]
""".strip(),
        encoding="utf-8",
    )

    blocks = load_mineru_blocks(data_dir, enable_image_ocr=False, enable_pdf_links=False)
    chunks = build_chunks(blocks)
    text_chunk = next(chunk for chunk in chunks if "更换墨盒" in chunk.text)

    assert text_chunk.metadata["related_images"] == ["images/step.jpg"]


def _sample_mineru_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "mineru"
    data_dir.mkdir()
    (data_dir / "demo_content_list.json").write_text(
        """
[
  {"type": "text", "text": "MinerU 是一个文档解析工具。", "page_idx": 0},
  {
    "type": "table",
    "table_body": "<table><tr><td>RAG</td><td>Qdrant</td></tr></table>",
    "table_caption": ["RAG 框架示例"],
    "page_idx": 0
  }
]
""".strip(),
        encoding="utf-8",
    )
    return data_dir

