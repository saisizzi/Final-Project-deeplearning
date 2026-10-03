import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useSimulation } from '../contexts/simulation-context';
import { ResearchChart } from '../components/ResearchChart';
import { loadBacktest, loadReplayEvents } from '../data/simulationApi';
import { BacktestPage } from './BacktestPage';
import { PopulationPage } from './PopulationPage';
import { ReplayPage } from './ReplayPage';

vi.mock('../contexts/simulation-context', () => ({ useSimulation: vi.fn() }));
vi.mock('../data/simulationApi', () => ({ loadBacktest: vi.fn(), loadReplayEvents: vi.fn() }));

const emptyDecision = {
  agent_id: 'A001',
  generation: 0,
  forecast: 0.25,
  decision_score: 0.31,
  threshold: 0.2,
  action: 'BUY',
  capital: 99950,
  position: 0,
  daily_pnl: -50,
  trade_count: 2,
  signal: 0.31,
  entry: { timestamp: '2024-01-08', action: 'BUY', price: 100, quantity: 10, realized_pnl: 0, reason: 'ENTRY' },
  exit: { timestamp: '2024-01-08', action: 'SELL', price: 95, quantity: 10, realized_pnl: -50, reason: 'END_OF_DAY' },
  trades: [],
};

const readyAgent = {
  agent_id: 'A001',
  generation: 1,
  status: 'READY',
  alive: false,
  starting_capital: 100000,
  ending_capital: null,
  capital: 100000,
  forecast: null,
  decision_score: null,
  threshold: 0.2,
  action: 'READY',
  position: 0,
  daily_pnl: 0,
  trade_count: 0,
  trade_history: [],
  equity_curve: [100000],
  parent_a: 'A001',
  parent_b: null,
  immigrant: false,
  elite: false,
};

const readyState = {
  experiment_id: 'local-evolution-development:SimpleRNN',
  data_split: 'evolution-development' as const,
  final_test_access: false as const,
  generation: 1,
  status: 'READY' as const,
  day: 0,
  generation_days: 10,
  date: null,
  counts: { BUY: 0, HOLD: 0, SELL: 0 },
  trade_count: 0,
  best_capital: 100000,
  mean_capital: 100000,
  worst_capital: 100000,
  agents: [readyAgent],
  generations: [
    { generation: 0, status: 'EVALUATED' as const, day: 10, metrics: { alive_count: 1, dead_count: 0, best_ending_capital: 101000, mean_ending_capital: 101000, worst_ending_capital: 101000 }, agents: [], daily_events: [] },
    { generation: 1, status: 'READY' as const, day: 0, metrics: null, agents: [readyAgent], daily_events: [] },
  ],
  daily_events: [],
  lineage: [],
  strategy_curves: { manual_baseline: [], buy_and_hold: [] },
  remaining_development_days: 100,
  population_config: {
    population_size: 1,
    starting_capital: 100000,
    survival_threshold: 0.9,
    mutation_rate: 0.05,
    mutation_sigma: 0.1,
    immigrant_fraction: 0.05,
    seed: 42,
  },
};

const controls = {
  state: readyState,
  experiments: [
    { experiment_id: 'local-evolution-development:SimpleRNN', model_name: 'SimpleRNN' as const, label: 'Evolution-development / SimpleRNN', data_split: 'evolution-development' as const, final_test_access: false as const },
    { experiment_id: 'local-evolution-development:LSTM', model_name: 'LSTM' as const, label: 'Evolution-development / LSTM', data_split: 'evolution-development' as const, final_test_access: false as const },
    { experiment_id: 'local-evolution-development:GRU', model_name: 'GRU' as const, label: 'Evolution-development / GRU', data_split: 'evolution-development' as const, final_test_access: false as const },
  ],
  loading: false,
  busy: false,
  error: null,
  selectedGeneration: 1,
  setSelectedGeneration: vi.fn(),
  selectExperiment: vi.fn().mockResolvedValue(undefined),
  nextDay: vi.fn().mockResolvedValue(undefined),
  runGeneration: vi.fn().mockResolvedValue(undefined),
  runGenerations: vi.fn().mockResolvedValue(undefined),
  reset: vi.fn().mockResolvedValue(undefined),
  refresh: vi.fn().mockResolvedValue(undefined),
  getGeneration: vi.fn(),
  getReplay: vi.fn(),
  getBacktest: vi.fn(),
};

describe('interactive research views', () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useSimulation).mockReturnValue(controls);
  });

  it('shows newborn agents as READY and keeps evaluation results unset', () => {
    render(<PopulationPage />);

    expect(screen.getAllByText('Not evaluated')).toHaveLength(2);
    expect(screen.getAllByText('READY').length).toBeGreaterThan(0);
    expect(screen.getByRole('option', { name: 'Generation 0 — EVALUATED' })).toBeTruthy();
    expect(screen.getByRole('option', { name: 'Generation 1 — READY' })).toBeTruthy();
    expect(screen.getByText('0 / 10')).toBeTruthy();
  });

  it('shows completed-generation deaths and actual next-generation mutation metrics', () => {
    const deadAgent = {
      ...readyAgent,
      generation: 0,
      status: 'DEAD',
      ending_capital: 90000,
      capital: 90000,
      dead_reason: 'no_profit',
      mutation_applied: false,
    };
    const evaluatedGeneration = {
      ...readyState.generations[0],
      metrics: {
        alive_count: 0,
        dead_count: 1,
        profit_making_parent_count: 0,
        offspring_count: 1,
        immigrant_count: 1,
        mutation_count: 0,
        best_ending_capital: 90000,
        mean_ending_capital: 90000,
        worst_ending_capital: 90000,
      },
      agents: [deadAgent],
    };
    vi.mocked(useSimulation).mockReturnValue({
      ...controls,
      selectedGeneration: 0,
      state: { ...readyState, generations: [evaluatedGeneration, readyState.generations[1]] },
    });

    render(<PopulationPage />);

    expect(screen.getByText('No profit by generation end')).toBeTruthy();
    expect(screen.getByText(/0 profit-making parents.*1 replacements.*1 immigrants/)).toBeTruthy();
    const mutationMetric = screen.getByText('Mutated next-gen agents').parentElement;
    expect(mutationMetric?.textContent).toBe('Mutated next-gen agents0');
  });

  it('labels legacy mutation records as unknown instead of claiming a false count', () => {
    const legacyAgent = { ...readyAgent, mutation_applied: null };
    vi.mocked(useSimulation).mockReturnValue({
      ...controls,
      state: {
        ...readyState,
        agents: [legacyAgent],
        generations: [
          { ...readyState.generations[0], metrics: { ...readyState.generations[0].metrics, mutation_count: null } },
          { ...readyState.generations[1], agents: [legacyAgent] },
        ],
      },
    });

    render(<PopulationPage />);

    expect(screen.getByText('Unknown (legacy)')).toBeTruthy();
    expect(screen.getByText('Mutation details for this generation were not recorded accurately by the previous tracking version.')).toBeTruthy();
    const mutationMetric = screen.getByText('Mutated agents this generation').parentElement;
    expect(mutationMetric?.textContent).toBe('Mutated agents this generationUnknown');
  });

  it('dispatches next-day, generation, N-generation, and reset controls', () => {
    render(<PopulationPage />);

    fireEvent.change(screen.getByLabelText('Experiment'), { target: { value: 'LSTM' } });
    fireEvent.click(screen.getByRole('button', { name: 'Next day' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run generation' }));
    fireEvent.change(screen.getByLabelText('Run generations'), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run 3' }));
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));

    expect(controls.nextDay).toHaveBeenCalledOnce();
    expect(controls.selectExperiment).toHaveBeenCalledWith('LSTM');
    expect(controls.runGeneration).toHaveBeenCalledOnce();
    expect(controls.runGenerations).toHaveBeenCalledWith(3);
    expect(controls.reset).toHaveBeenCalledOnce();
  });

  it('replays recorded events with previous and next day controls', async () => {
    const dayOne = { generation: 0, day: 1, date: '2024-01-08', counts: { BUY: 1, HOLD: 0, SELL: 0 }, trade_count: 2, best_capital: 99950, mean_capital: 99950, worst_capital: 99950, agents: [emptyDecision] };
    const dayTwo = { ...dayOne, day: 2, date: '2024-01-09', agents: [{ ...emptyDecision, action: 'HOLD', entry: null, exit: null, trades: [] }] };
    vi.mocked(loadReplayEvents).mockResolvedValue({ generation: 0, events: [dayOne, dayTwo] });
    vi.mocked(useSimulation).mockReturnValue({ ...controls, selectedGeneration: 0, state: { ...readyState, generation: 0, status: 'EVALUATED', generations: [{ ...readyState.generations[0], daily_events: [dayOne, dayTwo] }] } });

    render(<ReplayPage />);
    await waitFor(() => expect(screen.getByText('2024-01-08')).toBeTruthy());
    fireEvent.click(screen.getByRole('button', { name: 'Next day' }));
    expect(await screen.findByText('2024-01-09')).toBeTruthy();
    expect(loadReplayEvents).toHaveBeenCalledWith(0);
  });

  it('defaults Replay to the latest generation with recorded days, not a READY generation', async () => {
    const dayOne = { generation: 0, day: 1, date: '2024-01-08', counts: { BUY: 1, HOLD: 0, SELL: 0 }, trade_count: 2, best_capital: 99950, mean_capital: 99950, worst_capital: 99950, agents: [emptyDecision] };
    const state = { ...readyState, selectedGeneration: 1, generations: [
      { ...readyState.generations[0], daily_events: [dayOne] },
      readyState.generations[1],
    ] };
    vi.mocked(loadReplayEvents).mockResolvedValue({ generation: 0, events: [dayOne] });
    vi.mocked(useSimulation).mockReturnValue({ ...controls, selectedGeneration: 1, state });

    render(<ReplayPage />);

    expect(await screen.findByText('2024-01-08')).toBeTruthy();
    expect(loadReplayEvents).toHaveBeenCalledWith(0);
  });

  it('defaults Backtest to the latest evaluated generation when current generation is READY', async () => {
    vi.mocked(loadBacktest).mockResolvedValue({
      generation: 0,
      data_split: 'evolution-development',
      selected_agent: null,
      best_lineage: null,
      manual_baseline: [],
      buy_and_hold: [],
      final_test_enabled: false,
    });

    render(<BacktestPage />);

    await waitFor(() => expect(loadBacktest).toHaveBeenCalledWith(0, undefined));
    expect(screen.queryByRole('status')).toBeNull();
    expect(screen.getByText('Final held-out test')).toBeTruthy();
  });

  it('plots evaluated equity curves and lets users hide a strategy series', async () => {
    const evaluatedAgent = {
      ...readyAgent,
      generation: 0,
      status: 'ALIVE',
      alive: true,
      ending_capital: 101000,
      capital: 101000,
      equity_curve: [100000, 101000],
    };
    const completedGeneration = {
      ...readyState.generations[0],
      agents: [evaluatedAgent],
      daily_events: [{ generation: 0, day: 1, date: '2024-01-08', counts: { BUY: 1, HOLD: 0, SELL: 0 }, trade_count: 2, best_capital: 101000, mean_capital: 101000, worst_capital: 101000, agents: [emptyDecision] }],
    };
    vi.mocked(useSimulation).mockReturnValue({
      ...controls,
      state: { ...readyState, generations: [completedGeneration, readyState.generations[1]] },
    });
    vi.mocked(loadBacktest).mockResolvedValue({
      generation: 0,
      data_split: 'evolution-development',
      selected_agent: evaluatedAgent,
      best_lineage: evaluatedAgent,
      manual_baseline: [100000, 100500],
      buy_and_hold: [100000, 100800],
      final_test_enabled: false,
    });

    const { container } = render(<BacktestPage />);
    await waitFor(() => expect(container.querySelectorAll('.backtest-chart-section svg')).toHaveLength(1));
    const equityChart = container.querySelector('.backtest-chart-section svg');
    expect(equityChart?.querySelectorAll('polyline')).toHaveLength(4);

    const selectedSeries = container.querySelector('.chart-toggles input') as HTMLInputElement;
    fireEvent.click(selectedSeries);
    expect(equityChart?.querySelectorAll('polyline')).toHaveLength(3);
  });

  it('keeps fractional chart ranges proportional instead of implying a 100 percent swing', () => {
    render(<ResearchChart
      ariaLabel="Recorded drawdown"
      formatValue={(value) => `${(value * 100).toFixed(2)}%`}
      series={[{ name: 'Drawdown', values: [0, -0.001], tone: 'hold' }]}
    />);

    expect(screen.queryByText('100.00%')).toBeNull();
    expect(screen.queryByText('-100.00%')).toBeNull();
    expect(screen.getByText('-0.20%')).toBeTruthy();
  });

  it('renders finite chart points without emitting invalid SVG coordinates for missing values', () => {
    const { container } = render(<ResearchChart
      ariaLabel="Capital with a missing observation"
      series={[{ name: 'Capital', values: [100000, Number.NaN, 99000], tone: 'best' }]}
    />);

    expect(screen.getByRole('img', { name: 'Capital with a missing observation' })).toBeTruthy();
    expect(container.querySelectorAll('circle')).toHaveLength(2);
    expect(container.querySelector('svg')?.innerHTML).not.toContain('NaN');
  });
});