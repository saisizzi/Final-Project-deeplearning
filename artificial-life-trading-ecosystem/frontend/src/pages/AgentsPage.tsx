import { useState } from 'react';

import { formatCurrency, formatNumber, formatPercent } from '../components/format';
import { useSimulation } from '../contexts/simulation-context';
import type { RuntimeAgent, RuntimeDecision } from '../types/domain';

type AgentSortKey = 'agent_id' | 'status' | 'capital' | 'return' | 'trade_count' | 'score';

export function AgentsPage() {
  const { state, loading, selectedGeneration } = useSimulation();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [sortKey, setSortKey] = useState<AgentSortKey>('capital');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [selectedAgentId, setSelectedAgentId] = useState('');

  if (loading) return <section><h2>Agents</h2><p>Loading agent records...</p></section>;
  if (!state) return <section><h2>Agents</h2><p className="error-state">Interactive agent state is unavailable.</p></section>;

  const generation = state.generations.find((item) => item.generation === selectedGeneration);
  const allAgents = generation?.agents ?? [];
  const latest = new Map<string, RuntimeDecision>();
  for (const event of generation?.daily_events ?? []) for (const decision of event.agents) latest.set(decision.agent_id, decision);
  const visible = allAgents
    .filter((agent) => statusFilter === 'ALL' || agent.status === statusFilter)
    .filter((agent) => !search || `${agent.agent_id} ${agent.parent_a ?? ''} ${agent.parent_b ?? ''}`.toLowerCase().includes(search.toLowerCase()))
    .sort((left, right) => {
      const leftDecision = latest.get(left.agent_id);
      const rightDecision = latest.get(right.agent_id);
      const leftValues: Record<AgentSortKey, string | number> = {
        agent_id: left.agent_id,
        status: left.status,
        capital: left.ending_capital ?? left.capital,
        return: (left.capital - left.starting_capital) / left.starting_capital,
        trade_count: left.trade_count,
        score: leftDecision?.decision_score ?? -Infinity,
      };
      const rightValues: Record<AgentSortKey, string | number> = {
        agent_id: right.agent_id,
        status: right.status,
        capital: right.ending_capital ?? right.capital,
        return: (right.capital - right.starting_capital) / right.starting_capital,
        trade_count: right.trade_count,
        score: rightDecision?.decision_score ?? -Infinity,
      };
      const order = typeof leftValues[sortKey] === 'string'
        ? String(leftValues[sortKey]).localeCompare(String(rightValues[sortKey]))
        : Number(leftValues[sortKey]) - Number(rightValues[sortKey]);
      return sortDirection === 'asc' ? order : -order;
    });
  const selected = allAgents.find((agent) => agent.agent_id === selectedAgentId) ?? visible[0];
  const selectedDecision = selected ? latest.get(selected.agent_id) : undefined;

  function changeSort(key: AgentSortKey) {
    if (key === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(key);
      setSortDirection(key === 'agent_id' || key === 'status' ? 'asc' : 'desc');
    }
  }

  const sortButton = (key: AgentSortKey, label: string) => (
    <button type="button" onClick={() => changeSort(key)} aria-label={`Sort by ${label}`}>
      {label}{sortKey === key ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : ''}
    </button>
  );

  return (
    <>
      <section className="agent-page-head">
        <p className="section-kicker">Organism registry · generation {selectedGeneration}</p>
        <div className="section-heading-row"><h2>Agents</h2><span className="count-label">{visible.length} of {allAgents.length} organisms</span></div>
        <div className="field-row">
          <div><label htmlFor="agent-search">Find organism</label><input id="agent-search" type="search" placeholder="Agent or parent ID" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
          <div><label htmlFor="agent-status">Lifecycle state</label><select id="agent-status" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}><option value="ALL">All states</option><option value="READY">READY</option><option value="RUNNING">RUNNING</option><option value="ALIVE">ALIVE</option><option value="DEAD">DEAD</option></select></div>
        </div>
      </section>

      <section className="agent-table-section">
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr>
              <th>{sortButton('agent_id', 'Agent')}</th><th>{sortButton('status', 'State')}</th><th>{sortButton('capital', 'Capital')}</th><th>{sortButton('return', 'Return')}</th>
              <th>Forecast</th><th>{sortButton('score', 'Score')}</th><th>Threshold</th><th>Action</th><th>Position</th><th>Mutation</th><th>{sortButton('trade_count', 'Trades')}</th><th>Parent A</th><th>Parent B</th>
            </tr></thead>
            <tbody>
              {visible.map((agent: RuntimeAgent) => {
                const decision = latest.get(agent.agent_id);
                const returnValue = (agent.capital - agent.starting_capital) / agent.starting_capital;
                return (
                  <tr key={agent.agent_id} aria-selected={agent.agent_id === selected?.agent_id} onClick={() => setSelectedAgentId(agent.agent_id)}>
                    <td><button className="table-select agent-id" type="button" onClick={(event) => { event.stopPropagation(); setSelectedAgentId(agent.agent_id); }}>{agent.agent_id}</button></td>
                    <td><span className={`text-status status-${agent.status.toLowerCase()}`}>{agent.status}</span></td>
                    <td className="table-numeric">{formatCurrency(agent.ending_capital ?? agent.capital)}</td>
                    <td className="table-numeric">{agent.status === 'READY' ? 'Not evaluated' : formatPercent(returnValue, 2)}</td>
                    <td className="table-numeric">{decision ? formatNumber(decision.forecast, 5) : '—'}</td>
                    <td className="table-numeric">{decision ? formatNumber(decision.decision_score, 5) : '—'}</td>
                    <td className="table-numeric">{formatNumber(agent.threshold, 3)}</td><td>{decision?.action ?? 'READY'}</td>
                    <td className="table-numeric">{formatNumber(agent.position, 2)}</td><td>{agent.mutation_applied === null ? 'Unknown (legacy)' : agent.mutation_applied ? 'Applied' : '—'}</td><td className="table-numeric">{agent.trade_count}</td>
                    <td className="mono">{agent.parent_a ?? '—'}</td><td className="mono">{agent.parent_b ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {visible.length === 0 && <p className="empty-state">No organisms match these filters.</p>}
        </div>
      </section>

      <section className="selected-agent-detail">
        <p className="section-kicker">Selected organism</p>
        {selected ? <>
          <div className="section-heading-row"><h2 className="agent-id">{selected.agent_id}</h2><span className={`text-status status-${selected.status.toLowerCase()}`}>{selected.status}</span></div>
          <div className="selection-detail">
            <div><span className="metric-label">Generation</span><strong>{selected.generation}</strong></div>
            <div><span className="metric-label">Capital</span><strong>{formatCurrency(selected.ending_capital ?? selected.capital, 2)}</strong></div>
            <div><span className="metric-label">Trades</span><strong>{selected.trade_count}</strong></div>
            <div><span className="metric-label">Realized P&amp;L</span><strong>{formatCurrency(selected.trade_history.reduce((sum, trade) => sum + trade.realized_pnl, 0), 2)}</strong></div>
            <div><span className="metric-label">Latest decision</span><strong>{selectedDecision?.action ?? 'Not run'}</strong></div>
            <div><span className="metric-label">Parents</span><strong>{[selected.parent_a, selected.parent_b].filter(Boolean).join(' · ') || 'None recorded'}</strong></div>
          </div>
          {selected.dead_reason === 'no_profit' && <p className="error-state">This organism did not finish the generation above its starting capital. Its historical record is retained.</p>}
          {selected.mutation_applied === null && <p className="supporting-line">The previous tracking version did not record this organism&apos;s actual genome change.</p>}
          {selected.mutation_applied === true && <p className="supporting-line">This organism inherited a mutated genome from its profit-making parent(s).</p>}
        </> : <p className="empty-state">No organism is selected.</p>}
      </section>
    </>
  );
}