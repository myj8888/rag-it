# rag-it

一个面向“产品使用说明书”的本地 RAG 项目。

你把原始 PDF 放进 `data_origin/`，项目调用 MinerU 官方 API 解析 PDF，把解析结果整理到 `pdf_parse/`，然后为每一份 PDF 单独构建一套索引。问问题时，系统会先用大模型判断应该查哪些说明书，再去对应索引里检索和回答。

## 项目目录

```text
data_origin/          原始 PDF，例如打印机说明书、工具使用手册
pdf_parse/            MinerU 解析结果，每个说明书一个子目录
rag_storage/          每份说明书的索引、索引注册表、日志、MinerU 下载 zip
backend/              后端代码：MinerU 解析、RAG 构建、检索、问答
front/                前端代码：Streamlit 网页
tests/                自动化测试
docs/                 技术说明
```

后端代码按功能分层：

```text
backend/
  main.py             命令行入口
  config.py           配置读取
  models.py           通用数据结构
  mineru/             MinerU API、PDF 链接、图片 OCR
  rag/                切块、索引、路由、检索、问答
front/
  app.py              Streamlit 网页
```

## 一份说明书解析后的样子

```text
pdf_parse/canon_cp1300/
  origin.pdf
  content_list.json
  middle.json
  full.md
  images/
  manifest.json
  _raw_mineru/
```

`manifest.json` 可以手动改产品名、品牌、型号和描述。这个信息会用于“大模型路由”，也就是帮助系统判断问题应该查哪份说明书。

## 每份 PDF 的索引放在哪里

每份 PDF 单独一套索引：

```text
rag_storage/indexes/<doc_id>/
  bm25.sqlite3
  index_meta.json
```

向量索引在 Qdrant 里，不是项目目录里的文件。每份 PDF 对应一个 Qdrant collection：

```text
rag_<doc_id>
```

例如：

```text
rag_storage/indexes/canon_cp1300/bm25.sqlite3
Qdrant collection: rag_canon_cp1300
```

所有索引由这个文件统一登记：

```text
rag_storage/index_registry.json
```

## 第一次运行

1. 安装依赖：

```powershell
uv sync
```

2. 复制配置：

```powershell
Copy-Item .env.example .env
```

3. 打开 `.env`，至少填写：

```env
MINERU_API_TOKEN=你的 MinerU Token
LLM_API_KEY=你的 DeepSeek 或 OpenAI-compatible API Key
```

4. 启动 Qdrant：

```powershell
docker compose up -d
```

5. 确认 Ollama 里有 `bge-m3`：

```powershell
ollama list
```

如果没有：

```powershell
ollama pull bge-m3
```

## 跑通一份说明书

1. 把 PDF 放到 `data_origin/`：

```text
data_origin/canon_cp1300.pdf
```

2. 调用 MinerU 官方 API 解析：

```powershell
uv run rag parse data_origin/canon_cp1300.pdf
```

3. 构建索引：

```powershell
uv run rag index --doc canon_cp1300
```

也可以索引 `pdf_parse/` 下所有说明书：

```powershell
uv run rag index
```

4. 查看说明书和索引状态：

```powershell
uv run rag list-docs
```

5. 检索测试：

```powershell
uv run rag search "怎么更换墨盒"
```

不想让大模型路由，手动指定说明书：

```powershell
uv run rag search "怎么更换墨盒" --doc canon_cp1300
```

6. 问答：

```powershell
uv run rag ask "故障灯闪烁怎么办？"
```

7. 启动网页：

```powershell
uv run streamlit run front/app.py
```

## 常用命令

```powershell
uv run rag parse data_origin/canon_cp1300.pdf
uv run rag parse data_origin/canon_cp1300.pdf --index
uv run rag list-docs
uv run rag index
uv run rag index --doc canon_cp1300
uv run rag search "支持哪些纸张尺寸"
uv run rag search "支持哪些纸张尺寸" --doc canon_cp1300
uv run rag ask "如何排查无法连接 Wi-Fi？"
uv run streamlit run front/app.py
```

## `.env` 常用配置

```env
PDF_PARSE_DIR=pdf_parse
RAG_STORAGE_DIR=rag_storage
RAG_LOG_FILE=rag_storage/rag_it.log

MINERU_API_TOKEN=你的 MinerU Token
MINERU_MODEL_VERSION=vlm
MINERU_LANGUAGE=ch
MINERU_POLL_INTERVAL_SECONDS=10
MINERU_TIMEOUT_SECONDS=3600

IMAGE_OCR_ENABLED=true
PDF_LINK_EXTRACTION_ENABLED=true

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_PREFIX=rag

EMBEDDING_MODEL=bge-m3
OLLAMA_BASE_URL=http://localhost:11434

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=你的 API Key
LLM_MODEL=deepseek-v4-flash
```

## 现在支持什么

- 普通步骤问答：例如“怎么安装”“怎么更换耗材”。
- 故障排查：例如“错误灯闪烁怎么办”。
- 参数查询：例如“支持什么纸张尺寸”。
- 表格查询：MinerU 表格会转成可检索文本。
- 图片/按钮说明：图片块会进入索引；有本地图片时可以做 OCR。
- PDF 跳转链接提取：从 `origin.pdf` 中提取网页链接。
- 链接安全提示：对非 HTTPS、localhost/IP、punycode 域名等做基础风险提示。
- 多说明书路由：不指定 `--doc` 时，系统用大模型先判断要查哪些说明书。

链接安全检测只是基础规则，不等于专业反钓鱼或杀毒服务。真正打开陌生链接前仍然要人工确认域名是否可信。

## 测试

```powershell
uv run pytest
```
