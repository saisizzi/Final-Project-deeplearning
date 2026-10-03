import { useEffect, useState } from 'react';

import { ResearchChart, type ResearchChartSeries } from '../components/ResearchChart';
import { formatCurrency, formatNumber, formatPercent } from '../components/format';
import { useSimulation } from '../contexts/simulation-context';
import { loadBacktest } from '../data/simulationApi';
import type { RuntimeBacktest } from '../types/domain';

type CurveKey = 'selected' | 'lineage' | 'manual' | 'hold';

export function BacktestPage() {
  const { state, loading } = useSimulation();
  const [generationOverride, setGenerationOverride] = useState<number | null>(null);
  const [agentId, setAgentId] = useState('');
  const [backtest, setBacktest] = useState<{ key: string; value: RuntimeBacktest } | null>(null);
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const [visible, setVisible] = useState<Record<CurveKey, boolean>>({ selected: true, lineage: true, manual: true, hold: true });

  const evaluated = state?.generations.filter((generation) => generation.status === 'EVALUATED') ?? [];
  const latestEvaluated = evaluated.at(-1)?.generation;
  const selectedGeneration = generationOverride !== null && evaluated.some((item) => item.generation === generationOverride)
    ? generationOverride
    : latestEvaluated ?? state?.generation ?? 0;
  const selectedGenerationRecord = state?.generations.find((generation) => generation.generation === selectedGeneration);
  const selectedIsEvaluated = selectedGenerationRecord?.status === 'EVALUATED';
  const experimentId = state?.experiment_id;
  const requestKey = `${experimentId ?? ''}:${selectedGeneration}:${agentId}`;

  useEffect(() => {
    if (!selectedIsEvaluated) return undefined;
    let active = true;
    void loadBacktest(selectedGeneration, agentId || undefined)
      .then((result) => {
        if (active) {
          setBacktest({ key: requestKey, value: result });
          setError(null);
        }
      })
      .catch((requestError: unknown) => {
        if (active) {
          setError({
            key: requestKey,
            message: requestError instanceof Error ? requestError.message : 'Recorded curves are unavailable.',
          });
        }
      });
    return () => { active = false; };
  }, [requestKey, selectedGeneration, agentId, selectedIsEvaluated]);

  if (loading) return <section><h2>Backtest</h2><p>Loading recorded curves...</p></section>;
  if (!state) return <section><h2>Backtest</h2><p className="error-state">Interactive simulation is unavailable.</p></section>;

  const currentBacktest = selectedIsEvaluated && backtest?.key === requestKey ? backtest.value : null;
  const currentError = error?.key === requestKey ? error.message : null;
  const loadingBacktest = selectedIsEvaluated && currentBacktest === null && currentError === null;
  const curves: Array<{ key: CurveKey; name: string; tone: ResearchChartSeries['tone']; values: number[] }> = [
    { key: 'selected', name: 'Selected organism', tone: 'selected', values: currentBacktest?.selected_agent?.equity_curve ?? [] },
    { key: 'lineage', name: 'Best lineage', tone: 'best', values: currentBacktest?.best_lineage?.equity_curve ?? [] },
    { key: 'manual', name: 'Manual baseline', tone: 'manual', values: currentBacktest?.manual_baseline ?? [] },
    { key: 'hold', name: 'Buy-and-hold', tone: 'hold', values: currentBacktest?.buy_and_hold ?? [] },
  ];
  const selectedCurves = curves.filter((curve) => visible[curve.key] && curve.values.length > 0);
  const maxPoints = Math.max(0, ...curves.map((curve) => curve.values.length));
  const xLabels = Array.from({ length: maxPoints }, (_, index) => index === 0
    ? 'Start'
    : selectedGenerationRecord?.daily_events[index - 1]?.date.slice(0, 10) ?? `Day ${index}`);
  const drawdown = (values: number[]) => {
    let peak = -Infinity;
    return values.map((value) => {
      peak = Math.max(peak, value);
      return peak > 0 ? (value - peak) / peak : 0;
    });
  };
  const lineageCurve = currentBacktest?.best_lineage?.equity_curve ?? [];
  const lineageDrawdown = drawdown(lineageCurve);
  const selectedAgent = currentBacktest?.selected_agent;
  const selectedReturn = selectedAgent?.equity_curve.length && selectedAgent.equity_curve[0] > 0
    ? (selectedAgent.equity_curve.at(-1)! - selectedAgent.equity_curve[0]) / selectedAgent.equity_curve[0]
    : null;

  function toggleCurve(key: CurveKey) {
    setVisible((current) => ({ ...current, [key]: !current[key] }));
  }

  return (
    <>
      <section className="backtest-controls">
        {selectedIsEvaluated && currentError && <p className="error-state" role="status">{currentError}</p>}
        <div className="section-heading-row">
          <div><p className="section-kicker">Recorded results · evolution-development</p><h2>Strategy behavior</h2></div>
          <div className="field-row backtest-selectors">
            <div><label htmlFor="backtest-generation">Evaluated generation</label><select id="backtest-generation" value={selectedGeneration} onChange={(event) => { setGenerationOverride(Number(event.target.value)); setAgentId(''); }}>
              {evaluated.map((generation) => <option key={generation.generation} value={generation.generation}>Generation {generation.generation}</option>)}
              {evaluated.length === 0 && <option value={selectedGeneration}>No evaluated generation</option>}
            </select></div>
            <div><label htmlFor="backtest-agent">Selected organism</label><select id="backtest-agent" value={agentId || currentBacktest?.selected_agent?.agent_id || ''} onChange={(event) => setAgentId(event.target.value)} disabled={!selectedGenerationRecord || selectedGenerationRecord.status !== 'EVALUATED'}>
              {(selectedGenerationRecord?.agents ?? []).map((agent) => <option key={agent.agent_id} value={agent.agent_id}>{agent.agent_id}</option>)}
            </select></div>
          </div>
        </div>
        <div className="chart-toggles" aria-label="Visible strategy series">
          {curves.map((curve) => <label key={curve.key} className="series-toggle"><input type="checkbox" checked={visible[curve.key]} disabled={!curve.values.length} onChange={() => toggleCurve(curve.key)} /><i className={`legend-line ${curve.tone}`} /><span>{curve.name}</span></label>)}
        </div>
      </section>

      <section className="backtest-chart-section">
        <div className="section-heading-row"><div><p className="section-kicker">Portfolio value · INR</p><h2>Equity curve</h2></div><span className="count-label">{maxPoints ? `${maxPoints - 1} trading days` : 'No observations'}</span></div>
        {currentBacktest ? <ResearchChart ariaLabel="Saved development equity curves for selected organism, best lineage, manual baseline and buy-and-hold" xLabels={xLabels} formatValue={(value) => formatCurrency(value)} series={selectedCurves} /> : <div className="empty-state">{loadingBacktest ? 'Loading recorded curves…' : selectedIsEvaluated ? 'Recorded curves are unavailable for this generation.' : 'This generation has not been evaluated; no equity curve is available.'}</div>}
      </section>

      <section className="backtest-drawdown-section">
        <div className="section-heading-row"><div><p className="section-kicker">Risk trace</p><h2>Best-lineage drawdown</h2></div><span className="count-label">peak-to-trough · %</span></div>
        {lineageDrawdown.length ? <ResearchChart ariaLabel="Drawdown derived from saved best-lineage equity curve" xLabels={xLabels} formatValue={(value) => formatPercent(value, 2)} series={[{ name: 'Drawdown', values: lineageDrawdown, tone: 'hold' }]} /> : <div className="empty-state">No evaluated lineage curve is available.</div>}
      </section>

      <section className="backtest-metrics-section">
        <p className="section-kicker">Selected organism</p>
        <div className="metric-strip">
          <div><span className="metric-label">Ending capital</span><strong>{selectedAgent ? formatCurrency(selectedAgent.equity_curve.at(-1) ?? selectedAgent.capital, 2) : 'Not recorded'}</strong></div>
          <div><span className="metric-label">Return</span><strong>{selectedReturn === null ? 'Not recorded' : formatPercent(selectedReturn, 2)}</strong></div>
          <div><span className="metric-label">Recorded trades</span><strong>{selectedAgent?.trade_count ?? '—'}</strong></div>
          <div><span className="metric-label">Equity observations</span><strong>{selectedAgent?.equity_curve.length ?? 0}</strong></div>
        </div>
      </section>

      <section className="backtest-trades-section">
        <div className="section-heading-row"><div><p className="section-kicker">Execution record</p><h2>Selected organism trades</h2></div><span className="count-label">{selectedAgent?.trade_history.length ?? 0} events</span></div>
        {!selectedAgent?.trade_history.length ? <div className="empty-state">No recorded trades for the selected organism.</div> : (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Date</th><th>Action</th><th>Reason</th><th>Price</th><th>Quantity</th><th>Cost</th><th>Realized P&amp;L</th></tr></thead>
              <tbody>{selectedAgent.trade_history.map((trade, index) => <tr key={`${trade.timestamp}-${index}`}>
                <td className="mono">{trade.timestamp.slice(0, 10)}</td><td>{trade.action}</td><td>{trade.reason}</td>
                <td className="table-numeric">{formatCurrency(trade.price, 2)}</td><td className="table-numeric">{formatNumber(trade.quantity, 2)}</td>
                <td className="table-numeric">{trade.transaction_cost === undefined ? 'Not recorded' : formatCurrency(trade.transaction_cost, 2)}</td>
                <td className="table-numeric">{formatCurrency(trade.realized_pnl, 2)}</td>
              </tr>)}</tbody>
            </table>
          </div>
        )}
      </section>

      <section className="final-test-boundary">
        <div><p className="section-kicker">Separate evaluation boundary</p><h2>Final held-out test</h2><p>Final held-out evaluation has not been run. Development results above do not include final-test data.</p></div>
        <span className="status-label status-ready">NOT RUN</span>
      </section>
    </>
  );
}