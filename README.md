# pdf-ocr

스캔/스크린샷 PDF를 **로컬에서** OCR해 텍스트 + 도표/그림을 하나의 Markdown으로 합쳐주는 도구.
두 가지 백엔드를 지원합니다:

| 백엔드 | 무엇 | 언제 |
|--------|------|------|
| `vlm`    | 로컬 **Qwen3-VL-8B** (vLLM/llama.cpp/SGLang) | VLM 인프라를 이미 가지고 있고, 페이지 단위 자유도가 필요할 때 |
| `mineru` | OpenDataLab **MinerU** (DocLayout-YOLO + PP-OCRv5 + RapidTable) | 한국어/CJK 문서에서 **도표 영역을 깔끔히 잘라내야** 할 때 |

같은 입력으로 두 백엔드를 비교해본 결과, 한국어 스캔본의 figure/table 영역 정확도는 MinerU가 확실히 우위입니다.
대신 MinerU는 의존성이 크고 모델 다운로드가 필요합니다.

## 설치

```bash
# uv가 없다면
curl -LsSf https://astral.sh/uv/install.sh | sh

# 기본 (vlm 백엔드만)
uv sync

# MinerU 백엔드까지 포함
uv sync --extra mineru
# 또는 MinerU 공식 인덱스에서:
# uv pip install --extra-index-url https://wheels.myhloli.com 'mineru[core]'

# 개발 도구
uv sync --extra dev
```

## 사용 — MinerU 백엔드 (권장)

```bash
uv run pdf-ocr run input.pdf -o output/ --config configs/mineru.yaml
# 또는 --backend로 즉석 지정:
uv run pdf-ocr run input.pdf -o output/ -b mineru
```

첫 실행 시 MinerU가 모델 가중치를 받습니다(수 GB). 한국어 페이지는 `mineru.lang: korean` 설정 그대로 두면
됩니다. 출력은 다음과 같이 정돈됩니다:

```
output/
├── input.md
└── input_images/
    ├── <mineru가 추출한 figure/table 이미지들>
```

## 사용 — VLM 백엔드 (Qwen3-VL)

별도 터미널에서 vLLM 서버를 띄우고:

```bash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 2 \
  --port 8000
```

본체 실행:

```bash
cp .env.example .env   # LLM_BASE_URL / LLM_MODEL 확인
uv run pdf-ocr run input.pdf -o output/ --config configs/vllm.yaml
```

llama.cpp / SGLang은 `examples/run_local.sh` 참고.

## 구조

```
src/pdf_ocr/
├── cli.py                       # typer CLI (--backend 옵션)
├── pipeline.py                  # 백엔드 디스패처
├── backends/
│   ├── base.py                  # Backend / BackendResult
│   ├── vlm.py                   # Qwen3-VL via OpenAI-compat
│   └── mineru.py                # MinerU CLI 래퍼 + 출력 정규화
├── pdf/render.py                # pypdfium2로 PDF→PIL (vlm 백엔드 전용)
├── llm/                         # VLM 클라이언트, 프롬프트, JSON 스키마
├── ocr/                         # bbox crop, page extractor (vlm 백엔드 전용)
├── render/markdown.py           # PageLayout 리스트 → .md
└── utils/                       # logging, config 로더
```

## 트러블슈팅

### MinerU CLI를 못 찾는다고 함

`mineru` 바이너리가 PATH에 없습니다. 두 방법 중 하나:

```bash
uv pip install --extra-index-url https://wheels.myhloli.com 'mineru[core]'
# 또는 절대경로를 config에 명시
# configs/mineru.yaml
# mineru:
#   binary: /home/me/.venv/bin/mineru
```

### MinerU가 다른 백엔드로 떨어진다

```yaml
mineru:
  backend: pipeline           # 한국어는 pipeline이 가장 안전
  extra_args: ["-d", "cuda"]  # GPU 강제
```

### VLM 백엔드: `max_tokens > max_model_len`

vLLM 서버를 더 큰 컨텍스트로 다시 띄우세요. `--max-model-len 12288` 이상 권장.

### VLM 백엔드: 도표에 본문 텍스트가 섞여 들어옴

Qwen3-VL의 bbox 정밀도 한계입니다. 이 버전에서 padding을 제거하고 프롬프트를 강화했지만,
한국어 스캔본에서 깔끔한 figure crop이 중요하면 **`--backend mineru`로 전환**하는 것이 답입니다.

## 개발

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
