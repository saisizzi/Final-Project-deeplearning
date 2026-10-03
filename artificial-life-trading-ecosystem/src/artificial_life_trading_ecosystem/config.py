"""Project configuration utilities and application settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local env files
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables and defaults."""

    project_name: str = field(default_factory=lambda: os.getenv("PROJECT_NAME", "artificial-life-trading-ecosystem"))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    population_size: int = field(default_factory=lambda: int(os.getenv("POPULATION_SIZE", "100")))
    generations: int = field(default_factory=lambda: int(os.getenv("GENERATIONS", "50")))
    mutation_rate: float = field(default_factory=lambda: float(os.getenv("MUTATION_RATE", "0.05")))
    mutation_sigma: float = field(default_factory=lambda: float(os.getenv("MUTATION_SIGMA", "0.10")))
    starting_capital: float = field(default_factory=lambda: float(os.getenv("STARTING_CAPITAL", "100000.0")))
    survival_threshold: float = field(default_factory=lambda: float(os.getenv("SURVIVAL_THRESHOLD", "1.0")))
    immigrant_fraction: float = field(default_factory=lambda: float(os.getenv("IMMIGRANT_FRACTION", "0.05")))
    elite_count: int = field(default_factory=lambda: int(os.getenv("ELITE_COUNT", "1")))
    transaction_cost: float = field(default_factory=lambda: float(os.getenv("TRANSACTION_COST", "0.001")))
    forecast_window: int = field(default_factory=lambda: int(os.getenv("FORECAST_WINDOW", "30")))
    hidden_units: int = field(default_factory=lambda: int(os.getenv("HIDDEN_UNITS", "64")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    ticker: str = field(default_factory=lambda: os.getenv("TICKER", "^NSEI"))
    start_date: str = field(default_factory=lambda: os.getenv("START_DATE", "2020-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("END_DATE", "2024-12-31"))
    interval: str = field(default_factory=lambda: os.getenv("INTERVAL", "1d"))
    target_mode: str = field(default_factory=lambda: os.getenv("TARGET_MODE", "regression"))
    window_length: int = field(default_factory=lambda: int(os.getenv("WINDOW_LENGTH", "30")))
    train_proportion: float = field(default_factory=lambda: float(os.getenv("TRAIN_PROPORTION", "0.60")))
    validation_proportion: float = field(default_factory=lambda: float(os.getenv("VALIDATION_PROPORTION", "0.15")))
    evolution_proportion: float = field(default_factory=lambda: float(os.getenv("EVOLUTION_PROPORTION", "0.15")))
    test_proportion: float = field(default_factory=lambda: float(os.getenv("TEST_PROPORTION", "0.10")))
    epoch_length: int = field(default_factory=lambda: int(os.getenv("EPOCH_LENGTH", "20")))
    generation_trading_days: int = field(default_factory=lambda: int(os.getenv("GENERATION_TRADING_DAYS", "10")))
    scaler_type: str = field(default_factory=lambda: os.getenv("SCALER_TYPE", "StandardScaler"))
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "SimpleRNN"))
    dropout: float = field(default_factory=lambda: float(os.getenv("DROPOUT", "0.20")))
    optimizer: str = field(default_factory=lambda: os.getenv("OPTIMIZER", "adam"))
    loss: str = field(default_factory=lambda: os.getenv("LOSS", "mse"))
    epochs: int = field(default_factory=lambda: int(os.getenv("EPOCHS", "20")))
    early_stopping_patience: int = field(default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "5")))
    decision_hidden_units: int = field(default_factory=lambda: int(os.getenv("DECISION_HIDDEN_UNITS", "8")))
    decision_activation: str = field(default_factory=lambda: os.getenv("DECISION_ACTIVATION", "tanh"))
    context_length: int = field(default_factory=lambda: int(os.getenv("CONTEXT_LENGTH", "5")))
    signal_threshold: float = field(default_factory=lambda: float(os.getenv("SIGNAL_THRESHOLD", "0.20")))
    position_size: float = field(default_factory=lambda: float(os.getenv("POSITION_SIZE", "0.25")))
    stop_loss: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS", "0.03")))
    take_profit: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT", "0.06")))
    allow_short: bool = field(default_factory=lambda: os.getenv("ALLOW_SHORT", "false").lower() == "true")
    leverage: float = field(default_factory=lambda: float(os.getenv("LEVERAGE", "1.0")))
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    models_dir: Path = field(default_factory=lambda: ROOT_DIR / "models")
    results_dir: Path = field(default_factory=lambda: ROOT_DIR / "results")


settings = Settings()

__all__ = ["Settings", "settings", "ROOT_DIR"]
