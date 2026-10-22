# OCR options for look-n-see — research memo

**Date:** 2026-10-22 · **Prepared for:** adding OCR to the `look-n-see` MCP server (Python 3.10+, stdio, deps today: `mcp>=2`, `huggingface_hub>=0.24`, `httpx>=0.27`) and for distributing vision-only / OCR-only / both variants.

Every claim below carries an inline primary-source URL. Where a claim could not be verified from a primary source it is tagged **[UNVERIFIED]**. Pricing/latency figures are dated **as of 2026-10**. Grounding already established in-repo (not re-derived): look-n-see routes `openrouter:<slug>` → OpenRouter, `hf:<id>`/bare → HF inference, default model `openrouter:google/gemma-4-26b-a4b-it` (~$0.0002/call, ~2× lower median latency than the HF serverless route per the repo's own benchmark sweep in `README.md`), fallbacks `Qwen/Qwen3-VL-30B-A3B-Instruct` → `openrouter:google/gemma-4-31b-it`; scrape-n-search (`/mnt/shared/Source/scrape-n-search/README.md`) is the packaging precedent (single PyPI package, providers enabled by API-key presence, `ENABLED_TOOLS` allow-list).

---

# A. OCR engine options

## A1. VLM-as-OCR through the EXISTING providers (OpenRouter + HF chat models)

**Integration.** Zero new dependencies. look-n-see already speaks OpenAI-compatible chat-completions over httpx to both providers; an OCR tool is just a chat message with an image part + an OCR prompt on a vision model.

- OpenRouter's official multimodal docs: "Send images to vision-capable models for analysis, description, **OCR**, and more", via the standard `/api/v1/chat/completions` endpoint with an `image_url`-type content part, accepting plain URLs or base64 `data:...` URIs; PDFs are also accepted (`file` type) and OpenRouter's "intelligent PDF parsing system extracts text and handles both text-based and scanned documents" (https://openrouter.ai/docs/guides/overview/multimodal/overview).
- OpenRouter structured output: `response_format: {"type": "json_schema", "json_schema": {schema, strict: true}}`; strict mode guarantees schema-conforming JSON; **streaming with structured outputs is supported**; an optional "Response Healing" plugin post-processes invalid JSON; unsupported models fail the request with an explicit error (https://openrouter.ai/docs/guides/features/structured-outputs).
- HuggingFace chat-completions (Inference Providers, OpenAI-compatible): supports `image_url` parts, `response_format.type = json_schema` with `strict`, `tools`, `logprobs`, streaming; auth is `Authorization: Bearer hf_****` (token with Inference Providers permission) (https://huggingface.co/docs/inference-providers/en/tasks/chat-completion).

**Quality signals (official model cards, dated 2026-10):** general VLMs are strong on document text: Qwen2-VL-7B scores **DocVQA 94.5, OCRBench 845** (https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#evaluation); SmolVLM scores **DocVQA 81.6** at ~5 GB GPU RAM (https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct#evaluation). For *pure page-level OCR*, dedicated OCR VLMs beat general VLMs on olmOCR-bench: Surya's 650M model scores 83.3 while GOT-OCR (0.6B) scores 48.3 (https://github.com/datalab-to/surya — benchmark table), and PaddleOCR claims PP-OCRv6 (34.5M params) "surpasses mainstream VLMs (Qwen3-VL-235B, GPT-5.5)" on their benchmark suite (https://pypi.org/project/paddleocr/). PaddleOCR-VL-1.6 (0.9B doc-parsing VLM) reports 96.3% on OmniDocBench v1.6 (same page). **Verdict: for "read the text out of this image" in an agent loop, a small dedicated VLM/OCR model and a frontier VLM are both credible; VLM-as-OCR costs little to ship (no deps) and returns LLM-ready text, but pure-OCR benchmarks favor dedicated OCR models, and VLM output is free-form unless `response_format` constrains it.**

**Latency/cost.** Per-call cost is the vision model's input+output tokens (image ≈ 1k+ tokens). look-n-see's own dated benchmark: default OpenRouter route ≈ $0.0002/call and ~2× lower median latency vs the HF serverless route (`README.md`). Current model prices are on https://openrouter.ai/models (per-model, dated 2026-10). **Fit: yes** — it reuses the existing `VisionClient` seam verbatim; OCR is one more prompt per tool.

## A2. HuggingFace serverless Inference OCR models (TrOCR / PaddleOCR / GOT-OCR2.0 / Qwen2-VL / SmolVLM)

**Central finding (verified 2026-10-22): none of these are served by HF's first-party serverless endpoint.**

1. The official **hf-inference provider page lists every supported task**: ASR, feature extraction, fill-mask, image classification, image segmentation, object detection, question answering, summarization, table QA, text classification, text-to-image, token classification, translation, zero-shot classification. **There is no image-to-text, image-text-to-text, or OCR task at all**, and the page states "As of July 2025, hf-inference focuses mostly on CPU inference (e.g. embedding, text-ranking, text-classification, or smaller LLMs like BERT or GPT-2)" (https://huggingface.co/docs/inference-providers/en/providers/hf-inference).
2. Per-model provider mappings from the HF API (fetched 2026-10-22, `?expand[]=inferenceProviderMapping`): `microsoft/trocr-base-printed` → `{}`; `Qwen/Qwen2-VL-7B-Instruct` → `{}`; `HuggingFaceTB/SmolVLM-Instruct` → `{}`; `stepfun-ai/GOT-OCR2_0` → `{}` (i.e. registered with **no** inference provider). Only `Qwen/Qwen2.5-VL-72B-Instruct` has live mappings — `ovhcloud` (status "live") and `featherless-ai` (status "error") — i.e. VLMs under HF Inference Providers are routed to *third-party* providers, not HF's CPU serverless (https://huggingface.co/api/models/<id>?expand[]=inferenceProviderMapping).
3. The official `image-text-to-text` task page's provider mapping lists baseten, cerebras, cohere, deepinfra, featherless-ai, fireworks-ai, novita, nscale, ovhcloud, publicai, scaleway, together, zai-org — **not hf-inference** (https://huggingface.co/docs/inference-providers/en/tasks/image-text-to-text).
4. `PaddleOCR/PaddleOCR` on HF returns HTTP 401 from the models API (gated) (https://huggingface.co/api/models/PaddleOCR/PaddleOCR).
5. Licenses: TrOCR **MIT** (upstream https://github.com/microsoft/unilm/blob/master/LICENSE; model card https://huggingface.co/microsoft/trocr-base-printed); GOT-OCR2.0 **Apache-2.0** (https://huggingface.co/stepfun-ai/GOT-OCR2_0); Qwen2-VL **Apache-2.0** (https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct); SmolVLM **Apache-2.0** (https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct).
6. **`output_types`:** the Inference Providers chat-completion/image-text-to-text API has **no `output_types` parameter** — the request schema exposes OpenAI-style knobs (`response_format` `json_schema`/`json_object`, `tools`, `logprobs`, …) instead (https://huggingface.co/docs/inference-providers/en/tasks/chat-completion). (The legacy `output_types` feature page 404s on the current docs site.) Structured OCR output on HF is therefore done via `response_format`.
7. Model quality/latency context if run locally via Inference Endpoints: TrOCR is a small encoder-decoder for **single text-line images** (printed; intended use stated on its model card); GOT-OCR2.0 needs `torch==2.0.1` + `trust_remote_code` per its card (https://huggingface.co/stepfun-ai/GOT-OCR2_0#usage); SmolVLM ≈ 5 GB GPU RAM (card).

**Conclusion: there is no OCR model on the HF serverless endpoint as of 2026-10; the HF route for OCR is chat-completions against partner-hosted VLMs (A1). TrOCR/PaddleOCR/GOT-OCR2.0 are local/Inference-Endpoints-only.**

## A3. Local OCR engines (dependency weight / install size / license / quality)

All metadata below is from PyPI JSON metadata fetched 2026-10-22 (https://pypi.org/pypi/<pkg>/json). "Torch-free" is the property that matters for a dependency-light server.

| Engine | PyPI pkg (ver) | Runtime deps (requires_dist) | Python | License | Torch? | Notes |
|---|---|---|---|---|---|---|
| **Tesseract** + pytesseract | `pytesseract` 0.3.13 (4.4M wk downloads) | wrapper only; **system `tesseract` binary required** (Debian/Ubuntu apt `tesseract-ocr`, macOS brew) | >=3.8 | Apache-2.0 (wrapper: https://pypi.org/project/pytesseract/; engine: https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) | No | Classic LSTM OCR, 100+ langs via traineddata; ms-scale CPU; no bbox confidence in plain text output (use `image_to_data` TSV); **system dependency breaks pure-pip install** (https://pypi.org/project/pytesseract/) |
| **RapidOCR** | `rapidocr-onnxruntime` 1.4.4 (884K wk) | pyclipper, opencv-python, numpy, six, Shapely, PyYAML, Pillow, **onnxruntime>=1.7.0**, tqdm | <3.13,>=3.6 (3.10 ✓) | Apache-2.0 | **No** | PaddleOCR PP-OCR models converted to ONNX; 109-language PP-OCRv5-era models; fast CPU inference; returns boxes + confidence; ~tens of MB of models; the lightest fully-pip local option (https://pypi.org/project/rapidocr-onnxruntime/, https://github.com/RapidAI/RapidOCR) |
| **PaddleOCR** | `paddleocr` 3.7.0 (284K wk) | `paddlex[ocr-core]>=3.7.0`, aiohttp, PyYAML, requests… (extras: all/doc-parser/doc2md/ie/trans) | >=3.8 (badge 3.8–3.12) | Apache-2.0 | No (PaddlePaddle stack) | PP-OCRv6 (34.5M medium: +4.6% det/+5.1% rec vs PP-OCRv5, 5.2× CPU speedup, 0.13 s/page A100), PaddleOCR-VL-1.6 0.9B doc VLM (96.3% OmniDocBench v1.6), PP-StructureV3 → Markdown/JSON; 50–111 languages; **heavy footprint** (paddlepaddle runtime, model downloads) (https://pypi.org/project/paddleocr/) |
| **EasyOCR** | `easyocr` 1.7.2 (534K wk) | **torch, torchvision>=0.5**, opencv-python-headless, scipy, scikit-image, … | (no upper bound declared) | Apache-2.0 | **Torch** | 80+ languages; CRAFT detection + CRNN; returns boxes + confidence; CPU slow, GPU expected; weights auto-download (https://pypi.org/project/easyocr/) |
| **Surya** | `surya-ocr` 0.22.1 (31K wk) | **torch>=2.7, torchvision>=0.20, transformers>=5.12** + runtime backend **vllm (NVIDIA GPU) or llama.cpp (CPU/Apple)** — it auto-spawns an inference server | >=3.10 <4 | Code Apache-2.0; **weights modified OpenRAIL-M** (free for research/personal/startups <$5M funding/revenue; paid license beyond) | Torch + server | 650M doc-OCR VLM: **83.3 olmOCR-bench** ("top under 3B params"), 5.35 pages/s on RTX 5090, 91-language benchmark 87.2%, layout+reading order+tables; document-specialized, not scene text (https://pypi.org/project/surya-ocr/, https://github.com/datalab-to/surya) |
| **docTR** | `python-doctr` 1.1.0 (78K wk) | **torch<3,>=2.0, torchvision>=0.15**, onnx, opencv, pypdfium2… | **>=3.11** — NOT 3.10-compatible | Apache-2.0 | Torch | Detection+recognition two-stage (DBNet/FAST + CRNN/SAR/ViTSTR…), layout + table models; maintained by t2k GmbH (https://pypi.org/project/python-doctr/) |

**Fit for a dependency-light stdio MCP server:** RapidOCR is the only fully-pip, torch-free, Python-3.10 local engine (pytesseract is pip-able but needs the system `tesseract` binary). PaddleOCR/EasyOCR/Surya/docTR all drag multi-hundred-MB-to-GB runtimes; docTR additionally excludes 3.10; Surya's weights license restricts commercial use and it needs an inference server.

## A4. Dedicated cloud OCR APIs over plain httpx

All prices verified on official pricing pages **as of 2026-10-22**.

| API | Auth model (httpx-friendliness) | OCR pricing (2026-10) | Output structure | Notes |
|---|---|---|---|---|
| **Mistral OCR 4.1** (`/v1/ocr`, model `mistral-ocr-latest`) | **Bearer API key header** — trivially httpx-only; curl example in official docs: `-H "Authorization: Bearer ${MISTRAL_API_KEY}"` (https://docs.mistral.ai/studio/document-processing/basic_ocr) | **$4 /1,000 pages**; **$5 /1,000 annotated pages**; batch endpoint exists on the pricing page (50% batch discount is third-party-reported **[UNVERIFIED exact %, see https://docuocr.com/mistral-ocr-pricing]**) (https://docs.mistral.ai/inference/pricing, https://docs.mistral.ai/models/ocr-4-1) | Per-page **markdown**, images with bounding boxes, tables (`null`/markdown/html), hyperlinks, header/footer, page dimensions; optional `include_blocks` (paragraph-level boxes + labels + reading order, OCR 4+) and `confidence_scores_granularity` (page/block/word); 40+ languages (https://docs.mistral.ai/studio/document-processing/basic_ocr) | GA 2026-07-16, Premier tier, speed/performance 4/5 (model page). Inputs: image_url (PNG/JPEG/AVIF…) or document_url (PDF/PPTX/DOCX), URLs or base64 data URIs. **Best fit of the four for look-n-see's httpx-only constraint.** |
| **Google Cloud Vision** (Text/Document Text Detection) | Service-account OAuth2 access token (needs google-auth or manual JWT signing; plain API keys don't suit server billing) [**UNVERIFIED details** — see https://cloud.google.com/vision/docs/auth] | Text Detection & Document Text Detection: **first 1,000 units/mo free; $1.50/1,000 images** (1,001–5M); $0.60/1,000 after; per-image, PDF pages count as images (https://cloud.google.com/vision/pricing) | `TextAnnotation` per page with lines/words, **bounding boxes + per-word/line confidence** (https://cloud.google.com/vision/docs/ocr) | Cheap for images, per-feature billing. |
| **Google Document AI** (Enterprise OCR / Form Parser / Layout) | Same OAuth2 service-account story [**UNVERIFIED mechanics**] | Enterprise Document OCR: first 1,000 free, **$1.50/1,000 pages**, $0.60/1,000 after; OCR add-ons $6/1,000; Form Parser $30/1,000; Layout Parser $10/1,000 (https://cloud.google.com/document-ai/pricing) | Page/paragraph/word objects with text + normalized vertices + confidence | Document-focused sibling of Vision. |
| **Azure AI Document Intelligence** ("in Foundry Tools", 2026 naming) | Subscription key (`Ocp-Apim-Subscription-Key`) or Entra token [**UNVERIFIED header name** — see https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api] | Free tier 500 pages/mo; **Read: $1.50/1,000 pages** (0–1M, then $0.60); **all prebuilt (Layout, Document, Receipt, Invoice, ID, Contract…): $10/1,000 pages**; Custom extraction $30/1,000; Query fields $10/1,000; commitment tiers discount to ~$9.50–$8/1,000 prebuilt, $0.75–$0.53/1,000 Read (https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/ — JS-rendered values captured 2026-10-22) | `analyzeResult` with pages/lines/words, `boundingPolygon`, `confidence` per word/line, plus layout/tables/KV | Async analyze→getResult poll pattern; REST is httpx-simple once the key header is in. |
| **AWS Textract** | **AWS SigV4 IAM request signing** — needs botocore/boto3 or a hand-rolled SigV4 signer; worst fit for bare httpx (https://docs.aws.amazon.com/textract/latest/dg/sync-calling.html, https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html) | Detect Document Text: **$0.0015/page** (first 1M/mo), $0.0006/page after; Analyze Document: Tables $0.015, Forms $0.05, Queries $0.015, Signatures $0.0035 per page (first 1M); free tier 1,000 pages/mo (DDText) (https://aws.amazon.com/textract/pricing/) | `Blocks` (PAGE/LINE/WORD) with `Confidence`, `BoundingBox`, `Polygon`, relationships (verified sample in https://docs.aws.amazon.com/textract/latest/dg/sync-calling.html) | Sync single-page (JPEG/PNG/PDF/TIFF, base64 `Bytes` or S3), async for multi-page. |

---

# B. Distribution of the three variants (vision-only / OCR-only / both)

## B1. Single package + optional-dependencies extras

- **What extras are for (primary spec):** "An extra is an optional part of a distribution… each extra results in the declaration of additional dependencies of the distribution **when the extra is used** in a dependency specification"; multiple extras union; syntax `name[extra]` (https://packaging.python.org/en/latest/specifications/dependency-specifiers/ — the current home of the PEP 508 grammar).
- Declaring them: `[project.optional-dependencies]` (and console scripts via `[project.scripts]`) are both formal keys of the PEP 621 `[project]` table (https://packaging.python.org/en/latest/specifications/pyproject-toml/).
- Install forms:
  - `pip install 'look-n-see[ocr]'` (pip resolves the extra's deps; per PEP 508 semantics above).
  - `uvx` extras: official uv docs — "The `--from` option can be used to run a tool with extras: `uvx --from 'mypy[faster-cache,reports]' mypy …`", and `uvx` is the alias for `uv tool run` (https://docs.astral.sh/uv/guides/tools/). So: **`uvx --from 'look-n-see[ocr]' look-n-see`** (note: the extra goes inside `--from`, not a separate `--extra` flag).
  - Plain `uv tool install 'look-n-see[ocr]'` also works, and installs **all** executables the package ships (uv installs every entry point: "the following will install the `http`, `https`, and `httpie` executables" for `httpie`) (https://docs.astral.sh/uv/guides/tools/).
- **Precedent (strong):** Microsoft's MarkItDown ships ONE PyPI package (`markitdown`, MIT, console script `markitdown` via `[project.scripts]`) with ~12 feature extras, including the OCR backend `az-doc-intel = ["azure-ai-documentintelligence", "azure-identity"]` (https://github.com/microsoft/markitdown/blob/main/packages/markitdown/pyproject.toml).

## B2. Multiple console scripts in one package

- PEP 621 allows any number of entries in `[project.scripts]` (https://packaging.python.org/en/latest/specifications/pyproject-toml/), so one wheel can ship `look-n-see` and `look-n-see-ocr` (`"look_n_see.ocr_server:main"`), each registering a different tool set.
- uv tooling supports this natively: `uvx/httpie` example above shows package≠command mapping (`uvx --from httpie http`) and `uv tool install` installs every script (https://docs.astral.sh/uv/guides/tools/). Same for pip/pipx.
- If scripts differ only by role, the cleaner variant is B4 (one entry point, env-gated registration) — the second script adds a maintenance surface for little benefit unless the OCR role must be *unselectable* at startup.

## B3. Separate PyPI projects per variant

- Cost: every change must be released twice (or one project must stay in sync with the other); version numbers drift; CI + tag-triggered release pipelines (scrape-n-search's model, `mcp-server/pyproject.toml` + `.github/workflows/release.yml`, published as `scrape-n-search-mcp`) would each need duplication; consumers get confusing near-identical package names.
- **How mainstream MCP servers actually choose (verified 2026-10):** they ship ONE package per server and gate optional capability, not split per capability:
  - `firecrawl-mcp` (npm, MIT): one package exposing search/scrape/parse/crawl/map/interact/agent/jobs/paper-search — no per-capability packages (https://www.npmjs.com/package/firecrawl-mcp).
  - `@playwright/mcp` (npm): one package, one toolset (https://github.com/microsoft/playwright-mcp — README, single `npx @playwright/mcp@latest` install).
  - `scrape-n-search-mcp` (PyPI, the sibling repo): one package; providers/capabilities are enabled by API-key presence and an `ENABLED_TOOLS` allow-list at runtime (verified in `/mnt/shared/Source/scrape-n-search/README.md`).
  - MarkItDown: one package, feature extras (B1) — the closest analog to an OCR variant (https://github.com/microsoft/markitdown/blob/main/packages/markitdown/pyproject.toml).
- Conclusion: separate PyPI projects contradict both the maintenance economics (double release/versioning) and the established MCP-ecosystem norm. Only consider this if OCR deps must be *excluded from the wheel's metadata entirely* for a hard constraint (e.g. audit), which is not the case here.

## B4. Env-gated tool registration at startup

- Recognized pattern, directly sanctioned by the protocol: the MCP spec's `tools/list` is a server-side advertisement — "Servers that support tools MUST declare the `tools` capability" and return the tool list; nothing requires a fixed set, and "implementations are free to expose tools through any interface pattern that suits their needs"; `listChanged` exists for dynamic lists (https://modelcontextprotocol.io/specification/2025-06-18/server/tools).
- Working precedent in-repo: scrape-n-search's `ENABLED_TOOLS` ("Explicit tool allow-list… overrides the set derived from DEFAULT_PROVIDER") and `DEFAULT_PROVIDER` ("tools that provider cannot serve are hidden from the client") — verified in its README and implemented in `mcp-server/src/scrape_n_search/` (config drives tool registration at startup).
- For look-n-see this means: one entry point; at startup register `ocr_*` tools only when the OCR role is enabled (env `OCR_ENABLED=1` and/or `look-n-see-ocr` script wrapper); `tools/list` then advertises exactly the enabled surface per variant, which is what the MCP clients display. Combined with B1's extras, an OCR-only install = `pip install 'look-n-see[ocr]'` + `OCR_ENABLED=1`, a vision-only install = plain install, both = both flags — all from one PyPI release.

---

# C. Recommendation

Ship OCR in the existing httpx-only architecture, on the current single-package distribution model, with **three tiers**: (1) **zero-dependency VLM-as-OCR** as the default — a new `extract_text`/`ocr_image` tool that sends the existing image-normalization + an OCR prompt through the current `VisionClient` to OpenRouter (and HF chat-completions for partner-hosted VLMs — verified: HF's own serverless serves no OCR or VLM task as of 2026-10, https://huggingface.co/docs/inference-providers/en/providers/hf-inference), using `response_format.json_schema` (strict, supported by both providers) to force structured text+confidence output (https://openrouter.ai/docs/guides/features/structured-outputs, https://huggingface.co/docs/inference-providers/en/tasks/chat-completion); (2) **one fully-pip local engine, RapidOCR** (`rapidocr-onnxruntime`, Apache-2.0, onnxruntime-only — no torch, Python 3.10 ✓, boxes+confidence, verified from PyPI metadata) as the offline/privacy fallback, shipped as an optional dependency `[project.optional-dependencies] ocr = ["rapidocr-onnxruntime"]` — the exact pattern Microsoft uses for MarkItDown's `az-doc-intel` OCR extra (https://github.com/microsoft/markitdown/blob/main/packages/markitdown/pyproject.toml); and (3) **a plain-httpx cloud OCR provider behind a `MISTRAL_API_KEY`-gated adapter for document-grade accuracy** — Mistral OCR 4.1 (`POST https://api.mistral.ai/v1/ocr`, Bearer auth per official curl docs, $4/1,000 pages, per-page markdown + block bounding boxes + confidence, 40+ languages, https://docs.mistral.ai/studio/document-processing/basic_ocr, https://docs.mistral.ai/inference/pricing) — because Google/Azure need OAuth2/subscription-key ceremony [mechanics unverified] and Textract requires SigV4 signing, making Mistral the only one of the four that is trivially httpx-native. Distribution: keep **one PyPI project** (`look-n-see`) with the `ocr` extra and **one console script**, role selection by env-gated tool registration at startup (scrape-n-search's proven `ENABLED_TOOLS` pattern; sanctioned by the MCP spec's `tools/list`), so `pip install 'look-n-see[ocr]'`/`uvx --from 'look-n-see[ocr]' look-n-see` yields the OCR-only or both variant from a single release with zero version drift — do **not** create separate PyPI projects (double-release cost, against ecosystem precedent: firecrawl-mcp, @playwright/mcp, scrape-n-search-mcp). Excluded: EasyOCR/docTR/Surya as hard deps (torch/GB-scale footprints; docTR is Python >=3.11 only — incompatible with look-n-see's 3.10 floor; Surya's OpenRAIL-M weights restrict commercial use).

---

# Verified source URLs (all fetched 2026-10-22 unless noted)

**OpenRouter** (docs moved to /guides/ — old URLs 404):
- https://openrouter.ai/docs/guides/overview/multimodal/overview — images/PDFs, vision models, OCR mention, base64/URL input
- https://openrouter.ai/docs/guides/features/structured-outputs — json_schema strict, streaming, Response Healing

**HuggingFace / Inference Providers:**
- https://huggingface.co/docs/inference-providers/en/providers/hf-inference — hf-inference (serverless) task list; CPU focus note
- https://huggingface.co/docs/inference-providers/en/tasks/chat-completion — image_url, json_schema, auth; no output_types param
- https://huggingface.co/docs/inference-providers/en/tasks/image-text-to-text — VLM task, third-party provider mapping
- https://huggingface.co/api/models/{microsoft/trocr-base-printed, Qwen/Qwen2-VL-7B-Instruct, HuggingFaceTB/SmolVLM-Instruct, stepfun-ai/GOT-OCR2_0, Qwen/Qwen2.5-VL-72B-Instruct}?expand[]=inferenceProviderMapping — provider mappings ({} vs ovhcloud live)
- https://huggingface.co/microsoft/trocr-base-printed · https://huggingface.co/stepfun-ai/GOT-OCR2_0 · https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct · https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct — cards, licenses, DocVQA/OCRBench numbers
- https://github.com/microsoft/unilm/blob/master/LICENSE — TrOCR MIT
- https://huggingface.co/api/models/PaddleOCR/PaddleOCR — HTTP 401 (gated)

**Local engines (PyPI JSON metadata):**
- https://pypi.org/pypi/pytesseract/json · https://pypi.org/project/pytesseract/ — wrapper, apt dep, Apache-2.0
- https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE — engine Apache-2.0
- https://pypi.org/pypi/rapidocr-onnxruntime/json · https://pypi.org/project/rapidocr-onnxruntime/ — onnxruntime deps only, Apache-2.0, 3.6–3.12
- https://pypi.org/pypi/paddleocr/json · https://pypi.org/project/paddleocr/ — paddlex deps, PP-OCRv6/PaddleOCR-VL claims, extras
- https://pypi.org/pypi/easyocr/json · https://pypi.org/project/easyocr/ — torch+torchvision, Apache-2.0
- https://pypi.org/pypi/surya-ocr/json · https://pypi.org/project/surya-ocr/ · https://github.com/datalab-to/surya — torch+vllm/llama.cpp, OpenRAIL-M weights, olmOCR-bench 83.3
- https://pypi.org/pypi/python-doctr/json · https://pypi.org/project/python-doctr/ — torch, Python >=3.11, Apache-2.0

**Cloud OCR:**
- https://docs.mistral.ai/models/ocr-4-1 — OCR 4.1 features, /v1/ocr, prices ($4/$5 per 1k), GA date, speed 4/5
- https://docs.mistral.ai/inference/pricing — OCR 4.1 $4/1,000 pages, $0.4 cached, batch toggle
- https://docs.mistral.ai/studio/document-processing/basic_ocr — endpoint, Bearer curl, mistral-ocr-latest alias, markdown output, include_blocks, confidence granularity, 40+ languages
- https://cloud.google.com/vision/pricing — Text/Document Text Detection $1.50–$0.60/1,000
- https://cloud.google.com/document-ai/pricing — Enterprise OCR $1.50/1,000, Form Parser $30, Layout $10
- https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/ — Read $1.50, Prebuilt $10, Custom $30, free 500 pages/mo (JS-rendered capture)
- https://aws.amazon.com/textract/pricing/ — DDText $0.0015/page, Analyze pricing, free tiers (page updated 2026-10-15)
- https://docs.aws.amazon.com/textract/latest/dg/sync-calling.html — base64 Bytes/S3 input, Blocks sample (PAGE/LINE/WORD, Confidence, BoundingBox, Polygon)

**Packaging / distribution:**
- https://packaging.python.org/en/latest/specifications/dependency-specifiers/ — extras semantics (PEP 508 grammar)
- https://packaging.python.org/en/latest/specifications/pyproject-toml/ — [project] keys incl. optional-dependencies, scripts (PEP 621)
- https://docs.astral.sh/uv/guides/tools/ — uvx == uv tool run, `--from 'pkg[extra]'`, tool install installs all executables
- https://github.com/microsoft/markitdown/blob/main/packages/markitdown/pyproject.toml — extras incl. az-doc-intel, [project.scripts]
- https://www.npmjs.com/package/firecrawl-mcp — single-package MCP server precedent
- https://github.com/microsoft/playwright-mcp — single-package (npx @playwright/mcp) precedent
- https://modelcontextprotocol.io/specification/2025-06-18/server/tools — tools/list advertisement, listChanged
- Local: /mnt/shared/Source/look-n-see/README.md, /mnt/shared/Source/look-n-see/pyproject.toml, /mnt/shared/Source/scrape-n-search/README.md, /mnt/shared/Source/scrape-n-search/mcp-server/pyproject.toml — grounding (ENABLED_TOOLS/DEFAULT_PROVIDER hides tools; PyPI `scrape-n-search-mcp`; vision benchmark figures)

**[UNVERIFIED] items:** Mistral batch-exact-50% price (third-party sites claim $2/1k batch); Google Vision/Document-AI auth mechanics (service-account OAuth2) and Azure `Ocp-Apim-Subscription-Key` header name — cite Microsoft's REST quickstart; anything about model *quality* beyond the vendor-reported benchmarks above.

## Addendum (2026-10-22, implementation)

- `rapidocr-onnxruntime` **1.4.x requires Python `<3.13`** (PyPI metadata, verified during implementation); the `ocr` extra therefore uses environment markers: `>=1.4` on Python <3.13, `>=1.2` on >=3.13 (1.2.3 is the newest unconstrained release). RapidOCR's own project supports 3.14 only via the 1.2 line as of this date.
- HF `hf_inference`-mapped VLMs via `Qwen/Qwen2.5-VL-72B-Instruct` were **not** added to the fallback chain; OCR routes through the vlm engine, which reuses `VISION_FALLBACKS` unchanged.
