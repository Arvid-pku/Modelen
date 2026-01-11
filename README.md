# LLM Interpretability Workbench

Visualize and analyze transformer language model internals: attention patterns, logit lens, activation traces, and interventions.

## Usage Options

This project supports two usage modes:

1. **Local Development** - Clone/fork the repo and run the full backend + frontend locally for interactive model analysis and interventions
2. **Static Demo** - Visit the [GitHub Pages demo](https://arvid-pku.github.io/Modelen/) to explore pre-computed examples (no interventions support)

## Requirements

- Python 3.12+
- Node.js 18+
- CUDA GPU (recommended for larger models)

## Quick Start (Local Development)

### Backend Setup

**Option A: Using uv (Recommended)**

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS with Homebrew
brew install uv
```

Then run the backend:

```bash
cd backend
uv sync                    # Install dependencies
./run.sh                   # Start server (uses uv internally)
# Or manually:
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: Using pip + venv**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using conda**

```bash
conda create -n modelen python=3.12
conda activate modelen
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`

### Environment Variables

Configure the backend with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt2` | HuggingFace model to load |
| `DEVICE` | auto | Device to use (`cuda`, `cpu`, or auto-detect) |
| `DTYPE` | `float16` | Model dtype (`float16`, `float32`, `bfloat16`) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

Example:
```bash
MODEL_NAME=Qwen/Qwen2-0.5B DEVICE=cuda ./run.sh
```

## Features

- **Logit Lens**: Token predictions at each layer
- **Attention Maps**: Interactive heatmaps with zoom/pan
- **Activation Trace**: Hidden state norms across layers
- **Statistics**: Distribution analysis, layer correlations, anomaly detection
- **Interventions**: Skip layers, ablate heads, patch activations
- **Multi-model**: Switch models with optional 4-bit/8-bit quantization

## Supported Models

GPT-2, Qwen, LLaMA, Phi, Gemma, Mistral, OPT, BLOOM, GPT-Neo, or any HuggingFace causal LM.

```bash
# Examples
MODEL_NAME=gpt2 python main.py
MODEL_NAME=Qwen/Qwen2-0.5B python main.py
MODEL_NAME=meta-llama/Llama-3.2-1B python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model/info` | GET | Current model info |
| `/model/load` | POST | Load a model |
| `/inference` | POST | Run inference with analysis |
| `/comparison` | POST | Compare with/without interventions |
| `/memory` | GET | GPU memory usage |
| `/cache/clear` | POST | Clear model cache |



## License

MIT
