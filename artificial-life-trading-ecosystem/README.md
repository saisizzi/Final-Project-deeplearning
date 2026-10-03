# Artificial-Life Trading Ecosystem

This repository provides the foundational structure for an artificial-life trading ecosystem in which a shared, gradient-trained forecasting backbone supports a population of evolving trading agents.

## Core architectural principle

The recurrent forecasting model is shared and trained through gradient descent. The evolutionary genome is intentionally small and contains only the agent-specific decision head and trading parameters. The full forecasting backbone is not evolved.

## Repository layout

- `src/` contains the Python package and module skeletons.
- `configs/` stores project configuration and environment defaults.
- `data/` holds raw and processed datasets.
- `models/` stores forecasting and genome model artifacts.
- `app/` is reserved for the eventual dashboard or Streamlit app.
- `tests/` contains project validation and smoke tests.
- `docs/` contains architecture and design notes.

## Scope of this milestone

This repository foundation intentionally avoids implementing:

- data processing pipelines
- forecasting training logic
- trading strategies
- evolutionary algorithms
- real brokerage or market execution features

The focus here is a clean package layout, configuration system, utilities, and import-safe skeletons that can support Milestone 1 work.

## Quick start

Python 3.11 is the supported project environment. TensorFlow's forecasting tests are not available in the default Python 3.14 setup; running pytest there skips those tests. Use the Python 3.11 launcher explicitly on Windows:

```bash
cd artificial-life-trading-ecosystem
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

On macOS or Linux:

```bash
cd artificial-life-trading-ecosystem
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Confirm the active interpreter with `python --version`; it should report Python 3.11.x. Do not use Python 3.14 for the full suite because the TensorFlow-dependent forecast tests will be skipped.

## Run the interactive dashboard

Start the API from the repository root in one PowerShell terminal:

```powershell
$env:PYTHONPATH = ".\src"
python -m uvicorn artificial_life_trading_ecosystem.api.app:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd .\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173/`. The API docs are at `http://127.0.0.1:8000/docs`. The Population page can advance one day, run a generation, run several generations, switch the trained forecaster, and reset the selected simulation. At generation end, agents that finish at or below their starting capital are marked DEAD. Only agents that made a profit can parent the next generation; if none made a profit, new immigrants replenish the population. The frontend shows per-generation deaths, offspring parents, and actual mutated offspring counts. Death and lineage records remain available in historical views.

## Planned layers

- Data layer
- Forecasting layer
- Agent layer
- Trading simulation layer
- Evolutionary population layer
- Evaluation layer
- Lineage layer
- Dashboard layer
