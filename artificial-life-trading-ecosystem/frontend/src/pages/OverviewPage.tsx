import { Link } from 'react-router-dom';

import { ResearchChart } from '../components/ResearchChart';
import { formatCurrency } from '../components/format';
import { useExperiment } from '../contexts/experiment-context';
import { useSimulation } from '../contexts/simulation-context';

function numericMetric(metrics: Record<string, number | string | null> | null | undefined, key: string): number | null {
  if (!metrics) return null;
  const rawValue = metrics[key];
  if (rawValue == null) return null;
  const value = Number(rawValue);
  return Number.isFinite(value) ? value : null;
}

export function OverviewPage() {
  const experiment = useExperiment();
  const { state, loading, error } = useSimulation();

  if (loading) return <section><h2>Overview</h2><p>Loading selected experiment state...</p></section>;
  if (!state) return <section><h2>Overview</h2><p className="error-state">{error ?? 'Selected experiment state is unavailable.'}</p></section>;

  const evaluated = state.generations.filter((generation) => generation.status === 'EVALUATED');
  const latestEvaluated = evaluated.at(-1);
  const historical = evaluated.map((generation) => ({
    label: `G${generation.generation}`,
    best: numericMetric(generation.metrics, 'best_ending_capital') ?? 0,
    mean: numericMetric(generation.metrics, 'mean_ending_capital') ?? 0,
    median: numericMetric(generation.metrics, 'median_ending_capital') ?? 0,
  }));
  if (state.status === 'RUNNING' && state.day > 0) {
    historical.push({
      label: `G${state.generation} · D${state.day}`,
      best: state.best_capital,
      mean: state.mean_capital,
      median: state.mean_capital,
    });
  }

  const bestCapital = latestEvaluated ? numericMetric(latestEvaluated.metrics, 'best_ending_capital') : null;
  const meanCapital = latestEvaluated ? numericMetric(latestEvaluated.metrics, 'mean_ending_capital') : null;
  const alive = latestEvaluated ? numericMetric(latestEvaluated.metrics, 'alive_count') : null;
  const dead = latestEvaluated ? numericMetric(latestEvaluated.metrics, 'dead_count') : null;
  const bestAgent = latestEvaluated?.agents.reduce((best, agent) =>
    agent.ending_capital !== null && (!best || agent.ending_capital > (best.ending_capital ?? -Infinity)) ? agent : best, undefined as typeof latestEvaluated.agents[number] | undefined);

  return (
    <>
      <section className="lead-section overview-identity">
        <p className="section-kicker">Selected experiment · {state.data_split}</p>
        <div className="overview-identity-row">
          <div>
            <h2>{state.experiment_id.split(':').at(-1)} · {experiment.ticker}</h2>
            <p>Experiment ID {state.experiment_id} · {experiment.interval} · {experiment.target_mode} target · shared frozen backbone</p>
          </div>
          <span className={`status-label status-${state.status.toLowerCase()}`}>{state.status}</span>
        </div>
        <div className="identity-meta">
          <span>Data window <strong>{experiment.start_date} – {experiment.end_date}</strong></span>
          <span>Sequence <strong>{experiment.sequence_length} trading days</strong></span>
          <span>Population <strong>{experiment.population_size} organisms</strong></span>
        </div>
      </section>

      <section className="overview-primary">
        <div className="overview-lead-grid">
          <div className="dominant-metric">
            <span className="section-kicker">Best capital · generation {latestEvaluated?.generation ?? '—'}</span>
            <strong>{bestCapital === null ? 'Awaiting evaluation' : formatCurrency(bestCapital, 2)}</strong>
            <small>{latestEvaluated ? 'Recorded ending capital · evolution-development' : 'No evaluated generation in this experiment yet'}</small>
          </div>
          <dl className="overview-state-list">
            <div><dt>Active generation</dt><dd className="mono">{state.generation}</dd></div>
            <div><dt>Lifecycle</dt><dd>{state.status}</dd></div>
            <div><dt>Day in generation</dt><dd className="mono">{state.day} / {state.generation_days}</dd></div>
            <div><dt>Population survival</dt><dd>{alive === null ? 'Not evaluated' : `${alive} alive · ${dead ?? 0} dead`}</dd></div>
            <div><dt>Mean ending capital</dt><dd>{meanCapital === null ? 'Not evaluated' : formatCurrency(meanCapital, 2)}</dd></div>
          </dl>
        </div>
      </section>

      <section className="chart-section">
        <div className="section-heading-row">
          <div>
            <p className="section-kicker">Population trajectory</p>
            <h2>Capital across generations</h2>
          </div>
          <span className="count-label">INR · development only</span>
        </div>
        <ResearchChart
          ariaLabel="Best, mean, and median ending population capital across evaluated generations"
          xLabels={historical.map((generation) => generation.label)}
          formatValue={(value) => formatCurrency(value)}
          series={[
            { name: 'Best capital', values: historical.map((item) => item.best), tone: 'best' },
            { name: 'Mean capital', values: historical.map((item) => item.mean), tone: 'mean' },
            { name: 'Median capital', values: historical.map((item) => item.median), tone: 'median' },
          ]}
        />
      </section>

      <section className="overview-secondary">
        <div className="section-heading-row">
          <div><p className="section-kicker">Lineage</p><h2>Best evaluated organism</h2></div>
          <Link className="text-link" to="/lineage">Explore lineage</Link>
        </div>
        {bestAgent ? (
          <div className="lineage-summary-row">
            <strong className="agent-id">{bestAgent.agent_id}</strong>
            <span>Generation {bestAgent.generation}</span>
            <span>{bestAgent.status}</span>
            <span className="mono">{formatCurrency(bestAgent.ending_capital ?? 0, 2)}</span>
            <span>{bestAgent.trade_count} trades</span>
          </div>
        ) : <div className="empty-state">No evaluated lineage is available yet.</div>}
      </section>

      <section className="evaluation-status">
        <div>
          <p className="section-kicker">Evaluation boundary</p>
          <h2>Development results</h2>
          <p>Interactive evolution consumes only the evolution-development split. Historical trajectory values above are recorded generation outcomes.</p>
        </div>
        <div className="final-test-notice">
          <span className="field-label">Final held-out test</span>
          <strong>{experiment.final_test_status}</strong>
        </div>
      </section>
    </>
  );
}