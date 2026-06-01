"""Command line entrypoint.

Input: user CLI commands such as parse/index/search/ask.
Output: parsed files, per-document indexes, search results, or RAG answers.
Side effects: calls MinerU/Ollama/Qdrant/LLM services and writes local storage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from backend.config import Settings, load_settings
from backend.logging_config import configure_logging
from backend.mineru.api import MinerUClient, organize_mineru_zip
from backend.rag.chunking import build_chunks
from backend.rag.dense_index import DenseIndex
from backend.rag.doc_store import iter_parsed_documents
from backend.rag.embeddings import OllamaEmbeddingClient
from backend.rag.index_registry import (
    collection_name_for_doc,
    require_entries,
    sqlite_path_for_doc,
    upsert_entry,
)
from backend.rag.ingest import load_mineru_blocks
from backend.rag.pipeline import RagPipeline
from backend.rag.sparse_index import SparseIndex

app = typer.Typer(help="Product manual RAG over MinerU parsed PDF output.")


@app.command()
def index(
    doc: str | None = typer.Option(None, "--doc", help="Index only one parsed document folder under PDF_PARSE_DIR."),
) -> None:
    """Rebuild per-document sparse and dense indexes."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_file)
    documents = iter_parsed_documents(settings.data_dir, doc_id=doc)
    if not documents:
        target = settings.data_dir / doc if doc else settings.data_dir
        raise FileNotFoundError(f"No parsed documents found in {target}. Run `uv run rag parse <pdf>` first.")

    for document in documents:
        _rebuild_one_index(settings, document.doc_id, document.doc_dir)


@app.command("mineru-parse")
@app.command("parse")
def parse_pdf(
    pdf_path: Path,
    ocr: bool = typer.Option(False, "--ocr", help="Enable MinerU OCR during extraction."),
    run_index: bool = typer.Option(False, "--index", help="Rebuild this document index after parsing."),
) -> None:
    """Parse a local PDF with MinerU Precision API and prepare RAG inputs."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_file)
    if not settings.mineru_api_token:
        raise RuntimeError("MINERU_API_TOKEN is not configured. Fill it in .env before running parse.")

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    client = MinerUClient(
        token=settings.mineru_api_token,
        model_version=settings.mineru_model_version,
        language=settings.mineru_language,
        poll_interval_seconds=settings.mineru_poll_interval_seconds,
        timeout_seconds=settings.mineru_timeout_seconds,
    )
    download = client.parse_file(
        pdf_path=pdf_path,
        download_dir=settings.storage_dir / "mineru_downloads",
        is_ocr=ocr,
        on_progress=typer.echo,
    )
    artifacts = organize_mineru_zip(download.zip_path, pdf_path, settings.data_dir)

    doc_id = artifacts.content_list_path.parent.name
    typer.echo(f"Parsed document folder: {artifacts.content_list_path.parent}")
    typer.echo(f"MinerU content list ready: {artifacts.content_list_path}")
    if artifacts.markdown_path:
        typer.echo(f"Markdown ready: {artifacts.markdown_path}")
    if artifacts.middle_path:
        typer.echo(f"Middle/layout JSON ready: {artifacts.middle_path}")
    typer.echo(f"Origin PDF copied: {artifacts.origin_pdf_path}")
    typer.echo(f"Manifest ready: {artifacts.manifest_path}")

    if run_index:
        _rebuild_one_index(settings, doc_id, artifacts.content_list_path.parent)


@app.command("list-docs")
def list_docs() -> None:
    """List parsed documents and built index status."""
    settings = load_settings()
    documents = iter_parsed_documents(settings.data_dir)
    built = {entry.doc_id: entry for entry in require_entries(settings)}
    if not documents:
        typer.echo(f"No parsed documents found in {settings.data_dir}. Run `uv run rag parse <pdf>` first.")
        return
    for document in documents:
        entry = built.get(document.doc_id)
        status = f"indexed chunks={entry.chunk_count} collection={entry.qdrant_collection}" if entry else "not indexed"
        typer.echo(f"{document.doc_id}  {document.doc_dir}  {status}")


@app.command()
def search(
    query: str,
    doc: str | None = typer.Option(None, "--doc", help="Skip LLM routing and search one document index."),
) -> None:
    """Run hybrid retrieval. Without --doc, an LLM router chooses indexes first."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_file)
    response = RagPipeline(settings).search(query, doc_ids=[doc] if doc else None)
    _print_route(response.route.reasoning, response.route.selected_doc_ids, response.route.used_fallback)
    _print_results(response.sources)


@app.command()
def ask(
    query: str,
    doc: str | None = typer.Option(None, "--doc", help="Skip LLM routing and ask one document index."),
) -> None:
    """Run routed retrieval and generate an answer."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_file)
    answer = RagPipeline(settings).ask(query, doc_ids=[doc] if doc else None)
    _print_route(answer.route.reasoning, answer.route.selected_doc_ids, answer.route.used_fallback)
    typer.echo(answer.answer)
    typer.echo("\nSources:")
    _print_results(answer.sources)


def _rebuild_one_index(settings: Settings, doc_id: str, doc_dir: Path) -> None:
    blocks = load_mineru_blocks(
        doc_dir,
        enable_image_ocr=settings.image_ocr_enabled,
        enable_pdf_links=settings.pdf_link_extraction_enabled,
    )
    chunks = build_chunks(blocks)
    typer.echo(f"Loaded {len(blocks)} MinerU blocks from doc={doc_id} and built {len(chunks)} chunks.")

    sqlite_path = sqlite_path_for_doc(settings, doc_id)
    sparse_index = SparseIndex(sqlite_path)
    sparse_index.rebuild(chunks)
    typer.echo(f"SQLite BM25 index rebuilt at {sqlite_path}.")

    collection_name = collection_name_for_doc(settings, doc_id)
    embedding_client = OllamaEmbeddingClient(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )
    dense_index = DenseIndex(
        url=settings.qdrant_url,
        collection_name=collection_name,
        embedding_client=embedding_client,
        vector_size=settings.embedding_dim,
    )
    dense_index.rebuild(chunks)
    typer.echo(f"Qdrant collection '{collection_name}' rebuilt with {dense_index.count()} points.")

    document = iter_parsed_documents(settings.data_dir, doc_id=doc_id)[0]
    entry = upsert_entry(settings, document, chunk_count=len(chunks))
    typer.echo(f"Index registry updated: {settings.index_registry_path}")
    typer.echo(f"Index meta written: {settings.indexes_dir / entry.doc_id / 'index_meta.json'}")


def _print_route(reasoning: str, selected_doc_ids: list[str], used_fallback: bool) -> None:
    fallback = " fallback=true" if used_fallback else ""
    selected = ", ".join(selected_doc_ids) if selected_doc_ids else "none"
    typer.echo(f"Route:{fallback} selected=[{selected}] reason={reasoning}")


def _print_results(results) -> None:
    if not results:
        typer.echo("No results.")
        return
    for index, result in enumerate(results, start=1):
        page = result.page or "unknown"
        snippet = result.text.replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        typer.echo(
            f"{index}. doc={result.doc_id} page={page} type={result.block_type} "
            f"hybrid={result.hybrid_score:.4f} bm25={result.bm25_score:.4f} "
            f"vector={result.vector_score:.4f}"
        )
        typer.echo(f"   {snippet}")
        related_images = result.metadata.get("related_images") or []
        if related_images:
            typer.echo(f"   related_images: {', '.join(str(path) for path in related_images)}")


def main() -> None:
    try:
        app()
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
