import forecastComparisonJson from '../../../results/forecasts/comparison.json';

import type { ArtifactManifest, ExperimentMetadata, ForecastingResult, SimulationState } from '../types/domain';

const forecastComparison = forecastComparisonJson as Record<string, unknown>;

export function getArtifactManifest(state: SimulationState | null): ArtifactManifest {
  const evaluated = state?.generations.filter((generation) => generation.status === 'EVALUATED') ?? [];
  return {
    forecasting_metrics: Object.keys(forecastComparison).length > 0,
    generation_metrics: evaluated.length > 0,
    lineage_data: state?.lineage.some((record) => record.parent_ids.length > 0) ?? false,
    strategy_comparison: evaluated.some((generation) => Object.values(generation.strategy_curves ?? {}).some((curve) => curve.length > 0)),
    final_test: false,
    replay_data: state?.generations.some((generation) => generation.daily_events.length > 0) ?? false,
    experiment_metadata: state !== null,
  };
}

export function getExperimentMetadata(): ExperimentMetadata {
  const firstForecast = Object.values(forecastComparison)[0] as Record<string, unknown> | undefined;
  const featureNames = (firstForecast?.metadata as Record<string, unknown> | undefined)?.feature_names as string[] | undefined;

  return {
    experiment_id: 'local-evolution-development',
    experiment_name: 'Local evolution development',
    ticker: '^NSEI',
    start_date: '2020-01-01',
    end_date: '2024-12-31',
    interval: '1d',
    target_mode: 'regression',
    feature_set: featureNames ?? ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'high_low_range', 'open_close_range', 'rolling_volatility'],
    sequence_length: 30,
    forecasting_backbone: 'SimpleRNN',
    population_size: 10,
    generations: 0,
    seed: 42,
    starting_capital: 100000,
    survival_threshold: 0.9,
    mutation_rate: 0.05,
    mutation_sigma: 0.1,
    immigrant_fraction: 0.05,
    transaction_cost: 0.001,
    dataset_hash: null,
    experiment_status: 'development-only',
    final_test_status: 'Final held-out evaluation has not been run.',
    development_status: 'Interactive development simulation is available from the local API.',
    artifact_note: 'Forecasting results are loaded from the repository artifacts; population and simulation data are served by the local API.',
  };
}

export function loadForecastingResults(): ForecastingResult[] {
  return Object.entries(forecastComparison).map(([name, entry]) => {
    const data = entry as Record<string, unknown>;
    const metrics = data.metrics as Record<string, unknown> | undefined;
    const parameterCount = data.parameter_count as Record<string, unknown> | undefined;
    const trainingSeconds = data.training_duration_seconds as number | undefined;
    const history = data.history as Record<string, unknown> | undefined;

    return {
      name,
      parameter_count: Number(parameterCount?.total ?? 0),
      training_time_seconds: Number(trainingSeconds ?? 0),
      best_validation_loss: Number((metrics?.loss as number | undefined) ?? 0),
      validation_mae: Number((metrics?.mae as number | undefined) ?? 0),
      validation_rmse: Number((metrics?.rmse as number | undefined) ?? 0),
      epochs_trained: Number((history?.loss as unknown[] | undefined)?.length ?? 0),
    };
  });
}
