import { useEffect, useState } from 'react';

import { useSimulation } from '../contexts/simulation-context';
import { loadReplayEvents } from '../data/simulationApi';
import { formatCurrency, formatNumber } from '../components/format';
import type { RuntimeDailyEvent } from '../types/domain';

function money(value: number | null | undefined): string {
  return value === null || value === undefined ? 'Not recorded' : formatCurrency(value, 2);
}

export function ReplayPage() {
  const { state, loading } = useSimulation();
  const [events, setEvents] = useState<RuntimeDailyEvent[]>([]);
  const [eventIndex, setEventIndex] = useState(0);
  const [agentId, setAgentId] = useState('');
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationOverride, setGenerationOverride] = useState<number | null>(null);
  const [loadedGeneration, setLoadedGeneration] = useState<number | null>(null);
  const [loadedExperimentId, setLoadedExperimentId] = useState<string | null>(null);
  const replayable = state?.generations.filter((generation) => generation.daily_events.length > 0) ?? [];
  const latestReplayable = replayable.at(-1)?.generation;
  const replayGeneration = generationOverride !== null && replayable.some((generation) => generation.generation === generationOverride)
    ? generationOverride
    : latestReplayable ?? state?.generation ?? 0;
  const playbackEvents = replayable.length > 0
    && loadedGeneration === replayGeneration
    && loadedExperimentId === state?.experiment_id
    ? events
    : [];

  useEffect(() => {
    if (!replayable.length) return undefined;
    let active = true;
    void loadReplayEvents(replayGeneration)
      .then((result) => {
        if (active) {
          setEvents(result.events);
          setLoadedGeneration(replayGeneration);
          setLoadedExperimentId(state?.experiment_id ?? null);
          setEventIndex(0);
          setPlaying(false);
          setAgentId(result.events[0]?.agents[0]?.agent_id ?? '');
          setError(null);
        }
      })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : 'Recorded replay could not be loaded.');
      });
    return () => { active = false; };
  }, [replayGeneration, replayable.length, state?.experiment_id]);

  useEffect(() => {
    if (!playing || playbackEvents.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setEventIndex((current) => {
        if (current + 1 >= playbackEvents.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 700);
    return () => window.clearInterval(timer);
  }, [playing, playbackEvents.length]);

  if (loading) return <section><h2>Replay</h2><p>Loading saved simulation...</p></section>;
  if (!state) return <section><h2>Replay</h2><p>Interactive simulation is unavailable.</p></section>;

  const event = playbackEvents[eventIndex];
  const decision = event?.agents.find((item) => item.agent_id === agentId) ?? event?.agents[0];
  const showTrade = (trade: RuntimeDailyEvent['agents'][number]['entry']) => trade ? `${trade.action} ${money(trade.price)} (${trade.reason})` : 'None';

  return (
    <>
      <section className="replay-controls-section">
        <p className="section-kicker">Recorded organism memory · development only</p>
        <div className="field-row">
          <div>
            <label htmlFor="replay-generation">Generation</label>
            <select id="replay-generation" value={replayGeneration} onChange={(change) => setGenerationOverride(Number(change.target.value))}>
              {replayable.map((generation) => <option key={generation.generation} value={generation.generation}>Generation {generation.generation} · {generation.status}</option>)}
              {replayable.length === 0 && <option value={replayGeneration}>No recorded generation</option>}
            </select>
          </div>
          <div>
            <label htmlFor="replay-agent">Organism</label>
            <select id="replay-agent" value={agentId} onChange={(change) => setAgentId(change.target.value)} disabled={!event}>
              {(event?.agents ?? []).map((item) => <option key={item.agent_id} value={item.agent_id}>{item.agent_id}</option>)}
            </select>
          </div>
          <div className="replay-controls" aria-label="Replay controls">
            <button type="button" onClick={() => setEventIndex((current) => Math.max(0, current - 1))} disabled={!event || eventIndex === 0}>Previous day</button>
            <button type="button" onClick={() => setEventIndex((current) => Math.min(playbackEvents.length - 1, current + 1))} disabled={!event || eventIndex >= playbackEvents.length - 1}>Next day</button>
            <button type="button" onClick={() => setPlaying(true)} disabled={!event || playing || playbackEvents.length < 2}>Play</button>
            <button type="button" onClick={() => setPlaying(false)} disabled={!playing}>Pause</button>
          </div>
        </div>
        {error && <p role="alert">{error}</p>}
      </section>

      {!event || !decision ? <section><p>No saved daily events for this generation.</p></section> : (
        <>
          <section className="replay-session">
            <div className="replay-session-heading">
              <div><p className="section-kicker">Generation {event.generation} · day {event.day} of {state.generation_days}</p><h2>{event.date.slice(0, 10)}</h2></div>
              <div className="replay-action-block"><span className="metric-label">Decision</span><strong className="replay-action">{decision.action}</strong></div>
            </div>
            <div className="replay-readout-grid">
              <div><span className="metric-label">Forecast</span><strong className="mono">{formatNumber(decision.forecast, 6)}</strong></div>
              <div><span className="metric-label">Decision score</span><strong className="mono">{formatNumber(decision.decision_score, 6)}</strong></div>
              <div><span className="metric-label">Threshold</span><strong className="mono">{formatNumber(decision.threshold, 3)}</strong></div>
              <div><span className="metric-label">Position at close</span><strong className="mono">{formatNumber(decision.position, 2)}</strong></div>
              <div><span className="metric-label">Entry</span><strong>{showTrade(decision.entry)}</strong></div>
              <div><span className="metric-label">Exit</span><strong>{showTrade(decision.exit)}</strong></div>
              <div><span className="metric-label">Daily P&amp;L</span><strong>{money(decision.daily_pnl)}</strong></div>
              <div><span className="metric-label">Capital</span><strong>{money(decision.capital)}</strong></div>
            </div>
          </section>
          <section>
            <p className="section-kicker">Population activity · {event.date.slice(0, 10)}</p>
            <h2>Daily decisions</h2>
            <ol className="plain-list">
              {event.agents.map((item) => <li key={item.agent_id}><strong>{item.agent_id} {item.action}</strong><span>score {item.decision_score.toFixed(5)}</span><span>P&amp;L {money(item.daily_pnl)}</span><span>trades {item.trades.length}</span></li>)}
            </ol>
          </section>
        </>
      )}
    </>
  );
}