# rag-it 技术设计

## 目标

`rag-it` 面向产品使用说明书构建 RAG。产品可以是打印机、家电、设备，也可以是软件工具或平台文档。

主流程：

```text
原始 PDF -> MinerU 官方 API 解析 -> 统一解析目录 -> 每份 PDF 独立建索引 -> LLM 路由 -> 检索/问答/UI
```

## 目录结构

```text
backend/
  main.py
  config.py
  models.py
  logging_config.py
  mineru/
    api.py
    convert_middle_json.py
    image_ocr.py
    pdf_links.py
    link_safety.py
  rag/
    doc_store.py
    ingest.py
    chunking.py
    sparse_index.py
    dense_index.py
    embeddings.py
    retriever.py
    pipeline.py
    llm.py
    router.py
    index_registry.py
front/
  app.py
```

## 数据目录

```text
data_origin/          原始 PDF
pdf_parse/            MinerU 解析结果
rag_storage/          索引、索引注册表、日志、下载 zip
```

每份说明书：

```text
pdf_parse/<doc_id>/
  origin.pdf
  content_list.json
  middle.json
  full.md
  images/
  manifest.json
  _raw_mineru/
```

`manifest.json` 中的产品名、品牌、型号、描述会进入路由提示词。

## 独立索引

每份 PDF 单独构建 SQLite/BM25：

```text
rag_storage/indexes/<doc_id>/bm25.sqlite3
rag_storage/indexes/<doc_id>/index_meta.json
```

每份 PDF 单独使用 Qdrant collection：

```text
rag_<doc_id>
```

统一注册表：

```text
rag_storage/index_registry.json
```

注册表记录：

- `doc_id`
- 产品名、品牌、型号、描述
- 解析目录
- SQLite 路径
- Qdrant collection
- chunk 数量
- 更新时间

## 查询路由

`search` / `ask` 不带 `--doc` 时：

1. 读取 `index_registry.json`。
2. 将说明书列表和用户问题交给 LLM router。
3. LLM 返回 JSON，包含 `reasoning` 和 `selected_ids`。
4. 系统只检索被选中的说明书索引。
5. 如果 LLM 路由失败或 JSON 非法，兜底检索全部已建索引。
6. 如果 LLM 返回空列表，则提示没有匹配说明书。

带 `--doc` 时跳过路由：

```powershell
uv run rag ask "怎么连接 Wi-Fi？" --doc canon_cp1300
```

## 核心命令

```powershell
uv run rag parse data_origin/canon_cp1300.pdf
uv run rag index
uv run rag index --doc canon_cp1300
uv run rag list-docs
uv run rag search "怎么更换墨盒"
uv run rag ask "故障灯闪烁怎么办？"
uv run streamlit run front/app.py
```

## 测试范围

当前测试覆盖：

- MinerU API 上传、轮询、失败、超时、下载。
- MinerU zip 整理成统一解析目录。
- 缺少 `content_list.json` 时从 `layout.json` 兜底转换。
- 内容读取、chunk 构建、SQLite BM25 检索。
- 链接安全检测。
- 每个文档的索引路径、Qdrant collection 名、注册表写入。
- LLM router 正常 JSON 和非法 JSON 兜底。

运行：

```powershell
uv run pytest
```
