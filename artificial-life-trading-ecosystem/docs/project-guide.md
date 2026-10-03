# Artificial-Life Trading Ecosystem: Project Walkthrough

## 1. What the project does

This is a **historical-market trading simulator with an evolving population of trading agents**. It combines a trained time-series forecasting model with agents that use that forecast to make simulated BUY, HOLD, or SELL decisions. After each generation, performance determines which agents survive and can create the next generation.

The central design choice is that **the forecaster is shared and frozen during evolution**, while each agent has its own smaller, evolvable decision policy and trading parameters. In other words, the project evolves how agents *act on* a forecast; it does not evolve or retrain a separate forecasting model for every agent. That separation is described in [architecture.md](./architecture.md).

The project simulates trades against historical data. It does **not** connect to a broker, place real orders, or promise future profitability.

## 2. The system, from market data to a generation result

The main flow is:

1. **Load market data.** The data pipeline can load a local dataset or acquire historical data. The default ticker configured in [config.py](../src/artificial_life_trading_ecosystem/config.py) is `^NSEI`, with a daily interval and a configured date range.
2. **Validate and prepare it.** The pipeline checks the OHLC price columns, orders records by date, checks for duplicate dates and invalid price relationships, and derives features such as daily return, price range, and rolling volatility. Details are in [pipeline.py](../src/artificial_life_trading_ecosystem/data/pipeline.py).
3. **Get a forecast.** A trained recurrent model processes recent features and provides a forecast. Available architectures include SimpleRNN, LSTM, and GRU; saved model artifacts are under [results/forecasts](../results/forecasts).
4. **Let each agent make its own decision.** An agent combines the shared forecast with a short recent-market context and its own genome to calculate a decision score.
5. **Simulate trades and update capital.** The trading engine applies the agent’s action, position size, transaction costs, and risk limits, then records trade and equity history.
6. **Evaluate the generation.** After its trading days are complete, each agent’s ending capital is compared with its starting capital.
7. **Reproduce the next generation.** Profitable agents can parent offspring. The offspring inherit genomes through crossover and may receive actual mutations. The previous generation’s records remain available for inspection.

## 3. Forecasting and the agent “genome”

### Shared forecaster

The forecasting layer contains TensorFlow/Keras recurrent models. It is trained separately, using gradient descent, and then reused as a common forecasting backbone. Its shared interface and training, prediction, saving, and loading behavior are in [base_forecaster.py](../src/artificial_life_trading_ecosystem/models/forecasting/base_forecaster.py). The backend loads trained artifacts rather than training a new model every time the dashboard is opened.

The **Forecasting** page compares recorded model results such as parameter count, training time, validation loss, MAE, RMSE, and epochs. Those displayed comparison values come from the repository’s [forecast comparison data](../results/forecasts/comparison.json). Selecting a row on that page highlights its comparison; choosing the active model for interactive evolution is done through the model selector on the Population or Experiment page.

### Per-agent decision policy

Each agent has an individual [DecisionGenome](../src/artificial_life_trading_ecosystem/models/genomes/decision_genome.py). It contains the compact neural decision head and parameters such as:

- **Signal threshold:** how strong a score must be before producing a BUY or SELL signal.
- **Position size:** the fraction of available cash to use for a position.
- **Stop loss and take profit:** the loss or gain levels used to trigger an exit.
- **Transaction cost:** the per-trade cost used in accounting.
- **Decision-head weights:** values that turn the forecast and recent context into a score.

The [Agent](../src/artificial_life_trading_ecosystem/agents/agent.py) calculates a score between −1 and +1. A score at or above the threshold means BUY; at or below the negative threshold means SELL; a score between them means HOLD. A SELL signal does not open a short position: this is a long-only simulator.

## 4. How trading is simulated

The [TradingEngine](../src/artificial_life_trading_ecosystem/trading/engine.py) owns execution and accounting. It tracks cash, positions, trades, P&L, transaction costs, and equity history. The default settings include a 25% position size, a 0.20 signal threshold, a 0.1% transaction cost, a 3% stop loss, and a 6% take profit; actual agent values can vary as their genomes evolve.

In the **interactive dashboard runtime**, the simulation advances through historical daily bars. The decision is made using past information; a new long position can be opened at the day’s Open, and exposure is closed by that day’s end unless a stop-loss or take-profit condition closes it earlier. If a daily OHLC bar appears to cross both stop-loss and take-profit levels, the engine uses the conservative stop-loss-first rule because daily bars do not say which event happened first intraday. The project’s intended timing and leakage-prevention principles are also described in [trading_engine.md](./trading_engine.md).

This is a historical simulation, not a live market feed. Results depend on the chosen historical data, model artifacts, execution assumptions, costs, and random seed.

## 5. Generations, deaths, reproduction, and mutation

The interactive runtime runs generations of **10 trading days**. Its state progresses through these statuses:

- **READY:** a new generation is available but has not started.
- **RUNNING:** it is stepping through its trading days.
- **EVALUATED:** the generation’s results have been recorded and the next population has been produced.

The current survival rule is strict: an agent must finish the generation with **more capital than it started with**. An agent that finishes at break-even or at a loss is marked DEAD after the full generation; it is not eliminated halfway through the generation for temporarily losing money. Its historical trade, capital, and lineage records remain available.

Profitable agents are eligible to parent the next generation. The population can preserve an elite copy of a top performer and create additional offspring. With two parents, genomes can be combined through crossover; with only one profitable parent, asexual reproduction is available. Newborn agents start with fresh starting capital rather than inheriting the parent’s account balance.

Mutation is applied to offspring genomes. The **mutation rate** controls how often genome values are selected for perturbation, and **mutation sigma** controls the scale of the Gaussian perturbation. The UI’s mutation count refers to offspring whose genome actually changed due to mutation—it is not a count of individual genes changed. An elite copy is not mutated. If **no agents make a profit**, there are no profitable parents, so the next population is replenished with immigrants.

The interactive population logic lives in [population.py](../src/artificial_life_trading_ecosystem/evolution/population.py), and the daily generation lifecycle and saved records live in [simulation.py](../src/artificial_life_trading_ecosystem/runtime/simulation.py).

## 6. What each dashboard page is for

The app is a React/Vite frontend served by a FastAPI backend. The routes are wired in [App.tsx](../frontend/src/App.tsx); shared navigation and current experiment status are in [Layout.tsx](../frontend/src/components/Layout.tsx).

### Overview

The Overview page summarizes the selected experiment and latest evaluated generation: best and mean ending capital, alive/dead counts, the capital trajectory, and the best evaluated organism. It also distinguishes development results from the held-out test boundary. See [OverviewPage.tsx](../frontend/src/pages/OverviewPage.tsx).

### Forecasting

The Forecasting page compares recorded forecasting measurements for the available recurrent models and labels the backbone as frozen during evolution. It is primarily a **model comparison** view; the active interactive forecaster is selected elsewhere. See [ForecastingPage.tsx](../frontend/src/pages/ForecastingPage.tsx).

### Population

This is the main control and observation page. You can select a forecasting experiment, inspect a generation, advance one day, run a generation, run several generations, or reset the selected model’s interactive simulation. It shows daily BUY/HOLD/SELL counts, capital metrics, generation-by-generation capital curves, survival and death counts, offspring and parent summaries, and actual mutation counts.

For evaluated generations it also shows a dedicated death table with each dead agent’s reason, starting and ending capital, and profit or loss. During an unevaluated generation, the page indicates that death decisions are pending. The daily event log contains per-day agent decisions and trade summaries. See [PopulationPage.tsx](../frontend/src/pages/PopulationPage.tsx).

### Agents

The Agents page lets you search and filter organisms, filter by lifecycle state, and sort by fields such as capital, return, score, and trade count. Its table and detail panel show capital, forecast, decision score, threshold, current action, position, mutation status, trade history, and parent IDs. See [AgentsPage.tsx](../frontend/src/pages/AgentsPage.tsx).

### Lineage

The Lineage page visualizes ancestry and offspring for agents with recorded relationships. Select an organism to inspect its parents, ancestors, reproduction method, mutation status, offspring, ending capital, and death status. You can search/filter generation members and see parent and offspring links. See [LineagePage.tsx](../frontend/src/pages/LineagePage.tsx). The records themselves are represented by [LineageRecord](../src/artificial_life_trading_ecosystem/lineage/lineage.py).

### Backtest

The Backtest page examines an **already evaluated development generation**. It lets you choose an evaluated generation and organism, compare its equity curve with the best lineage, manual baseline, and buy-and-hold, and toggle those chart series. It also shows best-lineage drawdown, selected-agent return and trade count, and the recorded trade ledger. It does not run a held-out final test from the dashboard. See [BacktestPage.tsx](../frontend/src/pages/BacktestPage.tsx).

### Replay

The Replay page plays back saved daily events for a generation. You can select a generation and organism, step through days or play/pause, and inspect the forecast, decision score, threshold, action, entry, exit, daily P&L, capital, and population activity for that day. See [ReplayPage.tsx](../frontend/src/pages/ReplayPage.tsx).

### Experiment

The Experiment page shows experiment provenance and settings: selected model, ticker, interval, features, sequence length, seed, population size, generation length, starting capital, mutation settings, and artifact availability. It also provides a model selector. See [ExperimentPage.tsx](../frontend/src/pages/ExperimentPage.tsx).

## 7. How the backend and frontend communicate

The browser calls JSON endpoints exposed by [api/app.py](../src/artificial_life_trading_ecosystem/api/app.py). For example, the API loads the current simulation and experiments, advances a day, runs a generation or several generations, resets state, and serves generation, replay, and backtest data. The corresponding frontend calls are in [simulationApi.ts](../frontend/src/data/simulationApi.ts); shared loading, selection, and control state are managed by [SimulationProvider.tsx](../frontend/src/contexts/SimulationProvider.tsx).

Interactive state is saved separately by model under `results/interactive/`. The runtime writes state to disk so browser refreshes and API restarts can restore progress. The API restricts the interactive population to at most 10 agents, even though the general population configuration can specify a larger size.

## 8. Data boundaries and evaluation

The configured chronological partitions are training, validation, evolution-development, and final test. The interactive population uses the **evolution-development** data and its prior history context; it does not consume final-test rows. The dashboard and API report final-test access as disabled, and the current experiment status says that a final held-out evaluation has not been run.

There is also a separate offline experiment/evaluation layer for model and strategy measurements; its scope and artifact layout are described in [experiments.md](./experiments.md). Keeping final-test data separate matters because using it to choose models or evolve agents would make the final evaluation no longer independent.

## 9. Running the project

The current dashboard and API are served at:

- Dashboard: <http://127.0.0.1:5173/>
- API health: <http://127.0.0.1:8000/api/health>
- API documentation: <http://127.0.0.1:8000/docs>

To start them yourself, use two PowerShell terminals from the project folder:

```powershell
# Terminal 1: API
$env:PYTHONPATH = ".\src"
python -m uvicorn artificial_life_trading_ecosystem.api.app:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: frontend
Set-Location .\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

The root scripts include `prepare_data.py`, `train_forecasters.py`, `run_single_agent.py`, `run_population.py`, and `run_experiment.py`. Those are the entrypoints for the offline data, model, single-agent, population, and experiment workflows. Use each script’s `--help` to see its exact options.

## 10. A note about documentation and limitations

Some design documents describe the original planned architecture, and parts of the README still call the repository a foundation or refer to a future Streamlit dashboard. That does not fully describe the current implementation: the live dashboard is React/Vite and communicates with FastAPI. The frontend also reads the recorded forecasting comparison from the repository artifact, while the interactive population and its saved generations come from the API.

Finally, old saved generations can predate accurate mutation tracking. Those historical mutation values are shown as **Unknown (legacy)** rather than being presented as verified. The project’s measures and historical simulations are useful for experimentation and comparison, but they are not evidence that an evolved strategy will be profitable in future markets.
