# hf-vision-mcp — Project Brief

## Purpose
An MCP (Model Context Protocol) server that exposes a vision-capable model via
HuggingFace Inference. It lets models without native vision support "see": the
main model asks this server to analyze screenshots and explain images, and the
descriptions come back as tool results.

## Core capability
- Serve one or more vision models through HuggingFace Inference API
  (serverless or dedicated Inference Endpoints).
- Expose MCP tools such as:
  - `describe_image` — natural-language explanation of an image
  - `analyze_screenshot` — UI/state analysis of a screenshot (for tool-using agents)
  - generic `image_to_text` / Q&A over an image
- Return structured, machine-readable descriptions the main model can act on.

## Non-goals (tentative)
- No local model hosting / weights management.
- No training or fine-tuning.
- No image generation.

## Stack (tentative)
- Python 3.14 (only available toolchain on this workstation)
- MCP Python SDK (FastMCP)
- `huggingface_hub` Inference Client

## Status
Concept phase — brief will be refined after requirements Q&A.
