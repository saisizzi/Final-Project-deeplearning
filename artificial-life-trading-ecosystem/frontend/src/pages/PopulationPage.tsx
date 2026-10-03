import { useState } from 'react';

import { ResearchChart } from '../components/ResearchChart';
import { formatCurrency, formatNumber } from '../components/format';
import { useSimulation } from '../contexts/simulation-context';
import type { RuntimeAgent, RuntimeDailyEvent } from '../types/domain';

const generationOptions = [1, 3, 5, 10];

function latestDecisions(events: RuntimeDailyEvent[]): Map<string, RuntimeDailyEvent['agents'][number]> {
  const latest = new Map<string, RuntimeDailyEvent['agents'][number]>();
  for (const event of events) {
    for (const decision of event.agents) latest.set(decision.agent_id, decision);
  }
  return latest;
}

export function PopulationPage() {
  const { state, experiments, loading, busy, error, selectedGeneration, setSelectedGeneration, selectExperiment, nextDay, runGeneration, runGenerations, reset } = useSimulation();
  const [runCount, setRunCount] = useState(1);

  if (loading) return <section><h2>Population</h2><p>Loading saved simulation state...</p></section>;
  if (!state) return <section><h2>Population</h2><p className="error-state" role="alert">{error ?? 'Simulation state is unavailable.'}</p></section>;

  const selected = state.generations.find((generation) => generation.generation === selectedGeneration) ?? state.generations.at(-1);
  const metrics = selected?.metrics;
  const agents = selected?.agents ?? [];
  const selectedReady = selected?.status === 'READY';
  const selectedEvaluated = selected?.status === 'EVALUATED';
  const deadAgents = agents.filter((agent) => agent.status === 'DEAD');
  const mutationStatusUnknown = selectedEvaluated
    ? metrics?.mutation_count == null
    : agents.some((agent) => agent.mutation_applied === null);
  const mutationCount = mutationStatusUnknown
    ? 'Unknown'
    : selectedEvaluated
      ? Number(metrics?.mutation_count ?? 0)
      : agents.filter((agent) => agent.mutation_applied === true).length;
  const events = selected?.daily_events ?? [];
  const decisions = latestDecisions(events);
  const chartGenerations = state.generations.filter((generation) => generation.status === 'EVALUATED');
  const trajectory = chartGenerations.map((generation) => ({
    label: `G${generation.generation}`,
    best: Number(generation.metrics?.best_ending_capital ?? 0),
    mean: Number(generation.metrics?.mean_ending_capital ?? 0),
    worst: Number(generation.metrics?.worst_ending_capital ?? 0),
  }));
  if (state.status === 'RUNNING' && state.day > 0) {
    trajectory.push({ label: `G${state.generation} · D${state.day}`, best: state.best_capital, mean: state.mean_capital, worst: state.worst_capital });
  }
  const availableRunCount = Math.floor((state.remaining_development_days + state.day) / state.generation_days);
  const canRunGeneration = availableRunCount > 0;
  const lastEvent = events.at(-1);

  return (
    <>
      <section className="population-controls">
        {error && <p className="error-state" role="alert">{error}</p>}
        <div className="responsive-controls">
          <div>
            <label htmlFor="experiment-select">Experiment</label>
            <select id="experiment-select" value={state.experiment_id.split(':').at(-1) ?? 'SimpleRNN'} onChange={(event) => void selectExperiment(event.target.value as 'SimpleRNN' | 'LSTM' | 'GRU')} disabled={busy}>
              {experiments.map((experiment) => <option key={experiment.experiment_id} value={experiment.model_name}>{experiment.label}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor="generation-select">Inspect generation</label>
            <select id="generation-select" value={selectedGeneration} onChange={(event) => setSelectedGeneration(Number(event.target.value))}>
              {state.generations.map((generation) => <option key={generation.generation} value={generation.generation}>Generation {generation.generation} — {generation.status}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor="run-count">Run generations</label>
            <select id="run-count" value={runCount} onChange={(event) => setRunCount(Number(event.target.value))}>
              {generationOptions.map((count) => <option key={count} value={count}>{count} generation{count === 1 ? '' : 's'}</option>)}
            </select>
          </div>
          <div className="control-row" aria-label="Simulation controls">
            <button type="button" onClick={() => void nextDay()} disabled={busy || state.status === 'EVALUATED' || !canRunGeneration}>Next day</button>
            <button type="button" onClick={() => void runGeneration()} disabled={busy || state.status === 'EVALUATED' || !canRunGeneration}>Run generation</button>
            <button type="button" onClick={() => void runGenerations(runCount)} disabled={busy || availableRunCount < runCount}>Run {runCount}</button>
            <button type="button" onClick={() => void reset()} disabled={busy}>Reset</button>
            <span className="control-status" role="status" aria-live="polite">{busy ? 'Updating simulation...' : ''}</span>
          </div>
          <p className="supporting-line">At generation end, agents with ending capital at or below starting capital are marked DEAD. Only profit-making agents can parent the next generation; if none profit, the population is replenished with new immigrants.</p>
        </div>
      </section>

      <section className={`population-observation ${state.status === 'RUNNING' ? 'is-running' : ''}`} aria-live="polite">
        <div className="section-heading-row">
          <div>
            <p className="section-kicker">Population observation</p>
            <h2>Generation {state.generation} <span className={`status-label status-${state.status.toLowerCase()}`}>{state.status}</span></h2>
          </div>
          <div className="observation-day mono">Day {state.day} / {state.generation_days}<span>{lastEvent?.date ?? 'Not started'}</span></div>
        </div>
        <div className="action-counts" aria-label="Latest daily decisions">
          {(['BUY', 'HOLD', 'SELL'] as const).map((action) => <div key={action}><span>{action}</span><strong>{state.status === 'READY' ? 0 : state.counts[action]}</strong></div>)}
        </div>
        <div className="metric-strip population-capitals">
          <div><span className="metric-label">Best capital</span><strong>{formatCurrency(state.best_capital)}</strong></div>
          <div><span className="metric-label">Mean capital</span><strong>{formatCurrency(state.mean_capital)}</strong></div>
          <div><span className="metric-label">Worst capital</span><strong>{formatCurrency(state.worst_capital)}</strong></div>
          <div><span className="metric-label">Latest trades</span><strong>{state.status === 'READY' ? 0 : state.trade_count}</strong></div>
        </div>
      </section>

      <section className="chart-section population-trajectory">
        <div className="section-heading-row">
          <div><p className="section-kicker">Organisms gain · lose · reproduce</p><h2>Capital across generations</h2></div>
          <span className="count-label">₹ · end-of-generation values</span>
        </div>
        <ResearchChart
          ariaLabel="Population best, mean, and worst capital across evaluated generations and current running day"
          xLabels={trajectory.map((item) => item.label)}
          formatValue={(value) => formatCurrency(value)}
          series={[
            { name: 'Best', values: trajectory.map((item) => item.best), tone: 'best' },
            { name: 'Mean', values: trajectory.map((item) => item.mean), tone: 'mean' },
            { name: 'Worst', values: trajectory.map((item) => item.worst), tone: 'median' },
          ]}
        >
          <span className="chart-note">READY generations are shown as unevaluated; they do not add a result point.</span>
        </ResearchChart>
      </section>

      <section className="population-support">
        <div className="section-heading-row"><div><p className="section-kicker">Generation record</p><h2>Generation {selected?.generation ?? state.generation}</h2></div><span className={`status-label status-${(selected?.status ?? state.status).toLowerCase()}`}>{selected?.status ?? state.status}</span></div>
        <div className="metric-strip">
          <div><span className="metric-label">Day</span><strong>{selected?.day ?? 0} / {state.generation_days}</strong></div>
          <div><span className="metric-label">Ending capital</span><strong>{selectedEvaluated ? formatCurrency(Number(metrics?.best_ending_capital ?? state.best_capital)) : 'Not evaluated'}</strong></div>
          <div><span className="metric-label">Survival</span><strong>{selectedEvaluated ? `${Number(metrics?.alive_count ?? 0)} alive · ${Number(metrics?.dead_count ?? 0)} dead` : 'Not evaluated'}</strong></div>
          <div><span className="metric-label">Deaths this generation</span><strong>{selectedEvaluated ? Number(metrics?.dead_count ?? 0) : deadAgents.length}</strong></div>
          <div><span className="metric-label">Generation trades</span><strong>{selectedReady ? 0 : events.reduce((total, event) => total + event.trade_count, 0)}</strong></div>
          <div><span className="metric-label">Genetic diversity</span><strong>{metrics ? formatNumber(Number(metrics.genetic_diversity), 3) : '—'}</strong></div>
          <div><span className="metric-label">{selectedEvaluated ? 'Mutated next-gen agents' : 'Mutated agents this generation'}</span><strong>{mutationCount}</strong></div>
        </div>
        {metrics && <p className="supporting-line">{Number(metrics.profit_making_parent_count ?? metrics.alive_count ?? 0)} profit-making parents · {Number(metrics.offspring_count ?? 0)} replacements · {Number(metrics.immigrant_count ?? 0)} immigrants · {Number(metrics.sexual_children ?? 0)} sexual births · {Number(metrics.asexual_children ?? 0)} asexual births</p>}
      </section>

      <section className="agent-table-section">
        <div className="section-heading-row"><div><p className="section-kicker">Organism inspection</p><h2>Agents · generation {selected?.generation}</h2></div><span className="count-label">{agents.length} organisms</span></div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Agent</th><th>State</th><th>Forecast</th><th>Score</th><th>Threshold</th><th>Action</th><th>Position</th><th>Daily P&amp;L</th><th>Capital</th><th>Mutation</th><th>Trades</th></tr></thead>
            <tbody>
              {agents.map((agent: RuntimeAgent) => {
                const decision = decisions.get(agent.agent_id);
                return (
                  <tr key={agent.agent_id}>
                    <td className="agent-id">{agent.agent_id}</td><td><span className={`text-status status-${agent.status.toLowerCase()}`}>{agent.status}</span>{agent.dead_reason === 'no_profit' && <small className="quiet"> · no profit</small>}</td>
                    <td className="table-numeric">{decision ? formatNumber(decision.forecast, 5) : '—'}</td>
                    <td className="table-numeric">{decision ? formatNumber(decision.decision_score, 5) : '—'}</td>
                    <td className="table-numeric">{formatNumber(agent.threshold, 3)}</td><td>{decision?.action ?? (selectedReady ? 'READY' : '—')}</td>
                    <td className="table-numeric">{formatNumber(agent.position, 2)}</td>
                    <td className="table-numeric">{decision ? formatCurrency(decision.daily_pnl, 2) : '—'}</td>
                    <td className="table-numeric">{formatCurrency(agent.ending_capital ?? agent.capital)}</td><td>{agent.mutation_applied === null ? 'Unknown (legacy)' : agent.mutation_applied ? 'Applied' : '—'}</td><td className="table-numeric">{agent.trade_count}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {mutationStatusUnknown && <p className="supporting-line">Mutation details for this generation were not recorded accurately by the previous tracking version.</p>}

      <section className="death-record-section">
        <div className="section-heading-row"><div><p className="section-kicker">Generation mortality</p><h2>Deaths · generation {selected?.generation}</h2></div><span className="count-label">{selectedEvaluated ? deadAgents.length : 'Pending evaluation'}</span></div>
        {!selectedEvaluated ? <p className="empty-state">Deaths are determined after the full generation is evaluated against the break-even profit rule.</p> : deadAgents.length === 0 ? <p className="empty-state">No agents died; every agent finished above starting capital.</p> : (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Agent</th><th>Reason</th><th>Starting capital</th><th>Ending capital</th><th>Profit / loss</th></tr></thead>
              <tbody>{deadAgents.map((agent) => <tr key={agent.agent_id}>
                <td className="agent-id">{agent.agent_id}</td>
                <td>{agent.dead_reason === 'no_profit' ? 'No profit by generation end' : 'Did not meet profit rule'}</td>
                <td className="table-numeric">{formatCurrency(agent.starting_capital)}</td>
                <td className="table-numeric">{formatCurrency(agent.ending_capital ?? agent.capital)}</td>
                <td className="table-numeric">{formatCurrency((agent.ending_capital ?? agent.capital) - agent.starting_capital)}</td>
              </tr>)}</tbody>
            </table>
          </div>
        )}
      </section>

      <section className="event-log-section">
        <details>
          <summary>Daily event log <span className="mono">{events.length} days recorded</span></summary>
          {events.length === 0 ? <p className="empty-state">No daily events recorded for this generation.</p> : (
            <ol className="event-log">
              {events.map((event) => (
                <li key={`${event.generation}-${event.day}`}>
                  <strong>Day {event.day} · {event.date}</strong>
                  <div>{event.agents.map((decision) => <span key={decision.agent_id}><b>{decision.agent_id}</b> {decision.action}{decision.trades.length > 0 ? ` · ${decision.trades.map((trade) => `${trade.action} ${formatCurrency(trade.realized_pnl, 2)}`).join(', ')}` : ''} · {formatCurrency(decision.daily_pnl, 2)}</span>)}</div>
                </li>
              ))}
            </ol>
          )}
        </details>
      </section>
    </>
  );
}