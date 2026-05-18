# pdf-ocr

스캔/스크린샷 PDF를 로컬 VLM(**Qwen3-VL-8B-Instruct**)으로 OCR하고, 페이지 안의
**텍스트 / 도표 / 그림**을 구분해 하나의 Markdown 파일로 합쳐주는 도구입니다.

- 백엔드: OpenAI 호환 엔드포인트 (vLLM / llama.cpp / SGLang 무엇이든)
- 출력: `output/<pdf>.md` + `output/<pdf>_images/p001_fig01.png` …
- 환경: Python 3.11+ / **uv** 로 관리 / RTX 3090 Ti(24GB) 기준 검증

## 설치

```bash
# uv가 없다면
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 + 가상환경 생성
uv sync

# (개발 도구 포함)
uv sync --extra dev
```

## VLM 서버 띄우기 (예: vLLM)

```bash
# 별도 터미널에서
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

llama.cpp / SGLang은 `examples/run_local.sh` 참고.

## 사용

```bash
cp .env.example .env   # LLM_BASE_URL / LLM_MODEL 확인

# 단일 PDF 처리
uv run pdf-ocr run input.pdf -o output/

# 옵션
uv run pdf-ocr run input.pdf \
  -o output/ \
  --dpi 200 \
  --concurrency 4 \
  --config configs/vllm.yaml
```

결과:

```
output/
├── input.md
└── input_images/
    ├── p001_fig01.png
    ├── p002_tab01.png
    └── ...
```

## 구조

```
src/pdf_ocr/
├── cli.py                 # typer CLI
├── pipeline.py            # 페이지 루프 + 동시성
├── pdf/render.py          # PDF → PIL.Image
├── llm/openai_compat.py   # vLLM/llama.cpp/SGLang 공통 클라이언트
├── llm/schema.py          # PageLayout / Block (Pydantic)
├── llm/prompts.py         # Qwen3-VL용 OCR 프롬프트
├── ocr/extractor.py       # 페이지 이미지 → PageLayout(JSON)
├── ocr/crop.py            # bbox → PNG crop
├── render/markdown.py     # PageLayout 리스트 → .md
└── utils/                 # logging / retry / concurrency
```

## 트러블슈팅

### `BadRequestError: max_tokens=… cannot be greater than max_model_len=…`

vLLM/SGLang 서버의 컨텍스트 윈도우가 페이지 1장 + OCR 출력을 담기에 너무 작습니다.
서버를 충분한 컨텍스트로 다시 띄우세요:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \      # ← 이 값이 핵심
  --gpu-memory-utilization 0.85 \
  --port 8000
```

그리고 config의 `llm.max_tokens`가 `max-model-len - 입력토큰(이미지 1~2k + 프롬프트 ~1k)`
보다 작아야 합니다. 기본값 4096이면 대부분의 한국어 페이지를 담아냅니다. 페이지가 매우
조밀하다면 8192로 올려도 됩니다.

### 페이지가 잘려서 OCR이 중간에 끊김

`llm.max_tokens`를 늘리거나, `render.dpi`를 200 → 250으로 올려보세요. 단 DPI를 올리면
입력 토큰도 늘어나니 `max-model-len`도 함께 키워야 합니다.

### llama.cpp에서 빈 응답/JSON 깨짐

llama.cpp는 `response_format=json_schema`를 강제하지 못하므로 `configs/llamacpp.yaml`은
`use_response_format: false`로 설정돼 있습니다. 그래도 깨지면 `--grammar-file` 옵션으로
GBNF grammar를 강제하는 방법이 있습니다.

## 개발

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
