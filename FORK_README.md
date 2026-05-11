# ACE — Fernando's Fork

> Fork of [ai2cm/ace](https://github.com/ai2cm/ace) — the Ai2 Climate Emulator.
> Upstream `fme` package is **unchanged**. All additions live in `api/`, `agent/`, and `diagnostics/`.

---

## What this fork adds

The upstream ACE repository is a research-grade ML library for training and
evaluating atmospheric emulators. This fork builds an **application layer** on
top of it — turning the emulator into a queryable, validated, agent-accessible
service.

```
natural language question
        │
        ▼
┌───────────────────┐
│   LangGraph Agent  │  agent/
│  parse → call → interpret
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   FastAPI Layer    │  api/
│  /downscale        │
│  /health           │
│  /variables        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ACE Emulator      │  fme/ (upstream)
│  DiffusionModel    │
│  generate_numpy()  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Diagnostics       │  diagnostics/
│  non_negativity    │
│  value_ranges      │
│  spatial_smoothness│
│  wind_divergence   │
└───────────────────┘
```

---

## Motivation

ACE can simulate the atmosphere faster than any physics-based model. But three
gaps exist between "fast simulation" and "useful product":

1. **No API** — the emulator runs via CLI scripts, not queryable over HTTP.
2. **No natural language interface** — users need to know the internal variable
   names, grid conventions, and config format.
3. **No physical validation** — ML models can produce plausible-looking but
   physically inconsistent outputs (e.g. negative precipitation). Nothing in
   the upstream repo checks for this at inference time.

This fork addresses all three.

---

## Structure

```
ace/
├── fme/                        # upstream — do not modify
│   └── downscaling/            # diffusion-based downscaling model
├── api/                        # NEW — FastAPI application layer
│   ├── main.py                 # app entrypoint
│   ├── model.py                # singleton model loader + MockACEModel
│   ├── router.py               # HTTP endpoints
│   └── schemas.py              # Pydantic request/response models
├── agent/                      # NEW — LangGraph agent
│   ├── graph.py                # compiled StateGraph
│   ├── nodes.py                # parse_query, call_api, interpret_result
│   ├── run.py                  # CLI runner
│   └── state.py                # typed AgentState
├── diagnostics/                # NEW — physical plausibility checks
│   ├── checks.py               # individual check functions
│   └── report.py               # DiagnosticsReport aggregator
├── tests/                      # NEW — test suites
│   ├── test_api.py             # 6 tests — API layer
│   ├── test_agent.py           # 8 tests — agent (LLM + HTTP mocked)
│   └── test_diagnostics.py     # 16 tests — physical checks
├── notebooks/                  # NEW — exploration and demos
├── FORK_README.md              # this file
└── requirements-fork.txt       # fork-specific dependencies
```

---

## Quickstart

### 1. Install

```bash
git clone git@github.com:ferariz/ace.git
cd ace
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-fork.txt
```

### 2. Run the API (mock mode — no GPU, no checkpoint needed)

```bash
uvicorn api.main:app --reload --port 8000
```

Check it's live:

```bash
curl http://localhost:8000/api/v1/health | python3 -m json.tool
```

### 3. Run an inference call

```bash
curl -s -X POST http://localhost:8000/api/v1/downscale \
  -H "Content-Type: application/json" \
  -d '{
    "region": {"lat_min": -40, "lat_max": -20, "lon_min": -70, "lon_max": -45},
    "date": "2020-01-15T00:00",
    "variables": ["precipitation", "tmp2m"],
    "n_samples": 4
  }' | python3 -m json.tool
```

The response includes:
- Per-variable **mean and std grids** (fine resolution)
- A **diagnostics report** with physical plausibility checks

### 4. Run the agent

```bash
export OPENAI_API_KEY=your_key_here
python -m agent.run "What would precipitation look like over southern Brazil in January 2020?"
```

### 5. Run with a real checkpoint

Download a pretrained checkpoint from the
[ACE Hugging Face collection](https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4),
then:

```bash
CHECKPOINT_PATH=/path/to/checkpoint.pt uvicorn api.main:app --port 8000
```

The API and agent work identically — only the model backend changes.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Liveness check + model type |
| `GET`  | `/api/v1/variables` | Variables the loaded model produces |
| `POST` | `/api/v1/downscale` | Run downscaling inference |

### POST /api/v1/downscale

**Request:**
```json
{
  "region": {"lat_min": -40, "lat_max": -20, "lon_min": -70, "lon_max": -45},
  "date": "2020-01-15T00:00",
  "variables": ["precipitation", "tmp2m"],
  "n_samples": 4
}
```

**Response includes diagnostics on every call:**
```json
{
  "downscale_factor": 4,
  "variables": [...],
  "diagnostics": {
    "overall": "pass",
    "checks": [
      {"name": "non_negativity", "severity": "pass", ...},
      {"name": "value_ranges",   "severity": "pass", ...},
      {"name": "spatial_smoothness", "severity": "pass", ...},
      {"name": "wind_divergence", "severity": "pass", ...}
    ]
  }
}
```

---

## Diagnostics module

Every inference call runs four physical plausibility checks:

| Check | What it validates | Catches |
|-------|-------------------|---------|
| `non_negativity` | Precipitation ≥ 0 | Unphysical negative rain |
| `value_ranges` | All variables within known physical bounds | Extreme outliers, unit errors |
| `spatial_smoothness` | Gradient-to-value ratio below threshold | Checkerboard artifacts |
| `wind_divergence` | Mean \|∂u/∂x + ∂v/∂y\| below threshold | Unphysical flow patterns |

Severity levels: `pass` → `warn` → `fail`. The report is embedded in every
API response and can be used by the agent to flag quality issues.

> **Note:** `spatial_smoothness` threshold (2.0) is conservative and should be
> recalibrated against real ACE checkpoint output once GPU access is available.
> Current value is set for synthetic mock output.

---

## Tests

```bash
python -m pytest tests/ -v
```

| Suite | Tests | Dependencies |
|-------|-------|--------------|
| `test_api.py` | 6 | FastAPI TestClient only |
| `test_agent.py` | 8 | LLM + HTTP fully mocked |
| `test_diagnostics.py` | 16 | NumPy only |
| **Total** | **30** | **No GPU, no API keys, no server** |

---

## Design decisions

**MockACEModel over real checkpoint for development.**
The `RealACEModel` wrapper exists and works — but developing against a mock
that has no GPU dependency means the full API + agent + diagnostics stack is
runnable on any machine. Mock outputs use physically-constrained distributions
(log-normal precipitation, Gaussian temperature) rather than raw Gaussian noise.

**Protocol-based model abstraction.**
`api/model.py` defines `ACEModelProtocol` — swapping the real model for the
mock (or for a future different checkpoint) requires zero changes in the router.
Same principle applies to the LLM in `agent/nodes.py` — one function,
one swap.

**Diagnostics as a separate layer.**
Physical checks don't belong in the model or the router — they're a validation
concern. The `diagnostics/` module has zero dependencies on `fme` internals
and could be extracted as a standalone package or contributed upstream.

---

## Roadmap

- [ ] Notebook demo — end-to-end story from query to map visualization
- [ ] Real checkpoint integration — RunPod GPU instance
- [ ] Diagnostics calibration — threshold tuning on real ACE output
- [ ] Data loader — replace synthetic coarse inputs with real ERA5 data
- [ ] Diagnostics upstream PR — propose `fme/downscaling/diagnostics/` to ai2cm

---

## Upstream

This fork tracks [ai2cm/ace](https://github.com/ai2cm/ace) main branch.
To sync with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

All fork additions are isolated to `api/`, `agent/`, `diagnostics/`,
`tests/`, and `notebooks/` — upstream merges should never conflict.
