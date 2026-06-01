from __future__ import annotations

from backend.models import SearchResult
from backend.rag.retriever import deduplicate_results


def test_deduplicate_results_merges_same_pdf_link_url() -> None:
    results = [
        _result(
            chunk_id="link-1",
            text="PDF 跳转链接: https://arxiv.org/abs/2509.22186 锚点文本: MinerU Technical Report",
            block_type="pdf_link",
            hybrid_score=0.03,
        ),
        _result(
            chunk_id="link-2",
            text="PDF 跳转链接: https://arxiv.org/abs/2509.22186 锚点文本: MinerU Technical Report",
            block_type="pdf_link",
            hybrid_score=0.02,
        ),
        _result(
            chunk_id="link-3",
            text="PDF 跳转链接: https://arxiv.org/abs/2604.04771 锚点文本: MinerU2.5 Technical Report",
            block_type="pdf_link",
            hybrid_score=0.01,
        ),
    ]

    deduped = deduplicate_results(results)

    assert [result.chunk_id for result in deduped] == ["link-1", "link-3"]


def test_deduplicate_results_uses_url_metadata_when_available() -> None:
    results = [
        _result(
            chunk_id="link-1",
            text="Technical Report",
            block_type="pdf_link",
            metadata={"url": "https://arxiv.org/abs/2509.22186"},
        ),
        _result(
            chunk_id="link-2",
            text="MinerU Technical Report",
            block_type="pdf_link",
            metadata={"url": "https://arxiv.org/abs/2509.22186"},
        ),
    ]

    assert [result.chunk_id for result in deduplicate_results(results)] == ["link-1"]


def test_deduplicate_results_merges_same_text_on_same_page() -> None:
    results = [
        _result(chunk_id="text-1", text="放入墨盒 3 放入墨盒。", page=15),
        _result(chunk_id="text-2", text="放入墨盒   3\n放入墨盒。", page=15),
        _result(chunk_id="text-3", text="放入墨盒 3 放入墨盒。", page=16),
    ]

    deduped = deduplicate_results(results)

    assert [result.chunk_id for result in deduped] == ["text-1", "text-3"]


def _result(
    chunk_id: str,
    text: str,
    block_type: str = "text",
    page: int | None = 1,
    hybrid_score: float = 1.0,
    metadata: dict | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="MinerU_README_zh",
        text=text,
        page=page,
        block_type=block_type,
        source_path="pdf_parse/MinerU_README_zh/content_list.json",
        hybrid_score=hybrid_score,
        metadata=metadata or {},
    )
