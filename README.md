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

## 개발

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
