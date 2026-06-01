from pathlib import Path

from backend.rag.chunking import build_chunks
from backend.rag.sparse_index import SparseIndex
from backend.rag.ingest import load_mineru_blocks


def test_sparse_index_searches_chinese_terms(tmp_path: Path) -> None:
    data_dir = tmp_path / "mineru"
    data_dir.mkdir()
    (data_dir / "demo_content_list.json").write_text(
        """
[
  {"type": "text", "text": "MinerU 支持 Markdown 和 JSON。", "page_idx": 0},
  {
    "type": "table",
    "table_body": "<table><tr><td>RAG 框架</td><td>LangChain LlamaIndex RAGFlow</td></tr></table>",
    "page_idx": 0
  }
]
""".strip(),
        encoding="utf-8",
    )
    blocks = load_mineru_blocks(
        data_dir,
        enable_image_ocr=False,
        enable_pdf_links=False,
    )
    chunks = build_chunks(blocks)
    index = SparseIndex(tmp_path / "rag.sqlite3")
    index.rebuild(chunks)

    results = index.search("RAG框架有哪些", top_k=5)

    assert index.count() == len(chunks)
    assert results
    assert any("RAG" in result.text or "框架" in result.text for result in results)

