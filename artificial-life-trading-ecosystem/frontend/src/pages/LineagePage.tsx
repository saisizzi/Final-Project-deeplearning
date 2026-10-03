import { useState } from 'react';

import { formatCurrency, formatNumber } from '../components/format';
import { useSimulation } from '../contexts/simulation-context';
import type { RuntimeLineage } from '../types/domain';

function lineageAncestors(record: RuntimeLineage | undefined, byId: Map<string, RuntimeLineage>, limit = 3): RuntimeLineage[] {
  if (!record) return [];
  const result: RuntimeLineage[] = [];
  const queue = record.parent_ids.map((id) => ({ id, depth: 1 }));
  const seen = new Set<string>();
  while (queue.length && result.length < 12) {
    const next = queue.shift();
    if (!next || next.depth > limit || seen.has(next.id)) continue;
    seen.add(next.id);
    const ancestor = byId.get(next.id);
    if (!ancestor) continue;
    result.push(ancestor);
    queue.push(...ancestor.parent_ids.map((id) => ({ id, depth: next.depth + 1 })));
  }
  return result.sort((left, right) => left.generation - right.generation);
}

function stateClass(status: string): string {
  return status.toLowerCase();
}

export function LineagePage() {
  const { state, loading, selectedGeneration } = useSimulation();
  const [selectedAgentId, setSelectedAgentId] = useState('');
  const [filter, setFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');

  if (loading) return <section><h2>Lineage</h2><p>Loading lineage records...</p></section>;
  if (!state) return <section><h2>Lineage</h2><p className="error-state">Interactive lineage is unavailable.</p></section>;

  const generation = state.generations.find((item) => item.generation === selectedGeneration);
  const generationAgents = generation?.agents ?? [];
  const records = state.lineage.filter((record) => record.generation === selectedGeneration);
  const recordById = new Map(state.lineage.map((record) => [record.agent_id, record]));
  const isEvaluated = generation?.status === 'EVALUATED';
  const bestAgentId = isEvaluated
    ? generation?.best_agent_id ?? generationAgents.filter((agent) => agent.status === 'ALIVE').sort((left, right) => (right.ending_capital ?? right.capital) - (left.ending_capital ?? left.capital))[0]?.agent_id
    : undefined;
  const bestRecord = records.find((record) => record.agent_id === bestAgentId);
  const bestAgent = bestRecord ? generationAgents.find((agent) => agent.agent_id === bestRecord.agent_id) : undefined;
  const selectedRecord = records.find((record) => record.agent_id === selectedAgentId)
    ?? bestRecord
    ?? records[0];
  const currentAgent = selectedRecord ? generationAgents.find((agent) => agent.agent_id === selectedRecord.agent_id) : undefined;
  const ancestors = lineageAncestors(selectedRecord, recordById);
  const parentRecords = selectedRecord?.parent_ids.map((id) => recordById.get(id)).filter((record): record is RuntimeLineage => Boolean(record)) ?? [];
  const children = selectedRecord?.offspring_ids.map((id) => recordById.get(id)).filter((record): record is RuntimeLineage => Boolean(record)) ?? [];
  const filtered = records.filter((record) => {
    const matchesText = !filter || `${record.agent_id} ${record.parent_a ?? ''} ${record.parent_b ?? ''}`.toLowerCase().includes(filter.toLowerCase());
    return matchesText && (statusFilter === 'ALL' || record.status === statusFilter);
  });
  const ready = records.filter((record) => record.status === 'READY').length;
  const alive = records.filter((record) => record.status === 'ALIVE').length;
  const dead = records.filter((record) => record.status === 'DEAD').length;

  const renderNode = (record: RuntimeLineage, label: string, active = false) => {
    const agent = generationAgents.find((item) => item.agent_id === record.agent_id);
    return (
      <button
        key={`${label}-${record.agent_id}`}
        type="button"
        className={`lineage-node ${stateClass(record.status)}`}
        aria-pressed={active || selectedAgentId === record.agent_id}
        onClick={() => setSelectedAgentId(record.agent_id)}
        title={`${label}: ${record.agent_id}; ${record.status}; generation ${record.generation}`}
      >
        <span className="agent-id">{record.agent_id}</span>
        <small>{label} · {record.status}</small>
        {agent?.ending_capital !== null && agent?.ending_capital !== undefined && <small>{formatCurrency(agent.ending_capital)}</small>}
      </button>
    );
  };

  return (
    <>
      <section className="lead-section lineage-lead">
        <p className="section-kicker">Evolution record</p>
        <div className="section-heading-row">
          <div><h2>Best lineage</h2><p>Generation {selectedGeneration} · {generation?.status ?? 'Not recorded'}</p></div>
          <div className="lineage-state-counts">
            <span><b>{ready}</b> READY</span><span><b>{alive}</b> ALIVE</span><span><b>{dead}</b> DEAD</span>
          </div>
        </div>
        {isEvaluated && bestRecord ? (
          <div className="best-lineage-strip">
            {renderNode(bestRecord, 'Best evaluated organism', selectedRecord?.agent_id === bestRecord.agent_id)}
            <span className="lineage-best-meta">{bestAgent?.ending_capital === null || bestAgent?.ending_capital === undefined ? 'Not evaluated' : formatCurrency(bestAgent.ending_capital)} · {bestAgent?.trade_count ?? 0} trades</span>
          </div>
        ) : <div className="empty-state">{generation?.status === 'READY' ? 'This generation is READY; no best organism or survival outcome has been evaluated.' : 'No evaluated lineage is available for this generation.'}</div>}
      </section>

      <section className="lineage-explorer-section">
        <div className="section-heading-row">
          <div><p className="section-kicker">Selected ancestry</p><h2>{selectedRecord?.agent_id ?? 'No organism selected'}</h2></div>
          {selectedRecord && <span className={`text-status status-${stateClass(selectedRecord.status)}`}>{selectedRecord.status}</span>}
        </div>
        {!selectedRecord ? <div className="empty-state">No ancestry path has been recorded.</div> : (
          <div className="lineage-board" aria-label={`Lineage path for ${selectedRecord.agent_id}`}>
            <div className="lineage-stage">
              <span className="lineage-stage-label">Ancestors</span>
              <div className="lineage-nodes">{ancestors.length ? ancestors.slice(-6).map((record) => renderNode(record, `Generation ${record.generation}`)) : <span className="quiet">Root organism</span>}</div>
            </div>
            <div className="lineage-stage">
              <span className="lineage-stage-label">Parents</span>
              <div className="lineage-nodes">{parentRecords.length ? parentRecords.map((record) => renderNode(record, 'Parent')) : <span className="quiet">No parent recorded</span>}</div>
            </div>
            <div className="lineage-stage selected-stage">
              <span className="lineage-stage-label">Organism</span>
              <div className="lineage-nodes">{renderNode(selectedRecord, 'Selected', true)}</div>
            </div>
            <div className="lineage-stage">
              <span className="lineage-stage-label">Offspring</span>
              <div className="lineage-nodes">{children.length ? children.slice(0, 8).map((record) => renderNode(record, `Generation ${record.generation}`)) : <span className="quiet">No offspring recorded</span>}</div>
            </div>
          </div>
        )}
        {selectedRecord && (
          <dl className="selection-detail">
            <div><dt>Generation</dt><dd className="mono">{selectedRecord.generation}</dd></div>
            <div><dt>Parent A</dt><dd className="mono">{selectedRecord.parent_a ?? 'None recorded'}</dd></div>
            <div><dt>Parent B</dt><dd className="mono">{selectedRecord.parent_b ?? 'None recorded'}</dd></div>
            <div><dt>Reproduction</dt><dd>{selectedRecord.reproduction_method}</dd></div>
            <div><dt>Mutation</dt><dd>{selectedRecord.mutation_applied === null ? 'Unknown (legacy tracking)' : selectedRecord.mutation_applied ? 'Applied' : 'Not applied'}</dd></div>
            <div><dt>Immigrant</dt><dd>{selectedRecord.immigrant ? 'Yes' : 'No'}</dd></div>
            <div><dt>Death reason</dt><dd>{selectedRecord.status === 'DEAD' ? 'No profit by generation end' : '—'}</dd></div>
            <div><dt>Ending capital</dt><dd>{currentAgent?.ending_capital === null || currentAgent?.ending_capital === undefined ? 'Not evaluated' : formatCurrency(currentAgent.ending_capital, 2)}</dd></div>
            <div><dt>Return</dt><dd>{currentAgent?.ending_capital === null || currentAgent?.ending_capital === undefined ? 'Not evaluated' : formatNumber((currentAgent.ending_capital - currentAgent.starting_capital) / currentAgent.starting_capital * 100, 2) + '%'}</dd></div>
          </dl>
        )}
      </section>

      <section className="lineage-list-section">
        <div className="section-heading-row"><div><p className="section-kicker">Generation members</p><h2>Explore organisms</h2></div><span className="count-label">Showing {filtered.length} of {records.length}</span></div>
        <div className="field-row lineage-filters">
          <div><label htmlFor="lineage-filter">Find organism</label><input id="lineage-filter" type="search" placeholder="Agent or parent ID" value={filter} onChange={(event) => setFilter(event.target.value)} /></div>
          <div><label htmlFor="lineage-status">Lifecycle state</label><select id="lineage-status" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}><option value="ALL">All states</option><option value="READY">READY</option><option value="ALIVE">ALIVE</option><option value="DEAD">DEAD</option></select></div>
        </div>
        <div className="table-scroll lineage-table">
          <table>
            <thead><tr><th>Organism</th><th>State</th><th>Parent A</th><th>Parent B</th><th>Method</th><th>Mutation</th><th>Offspring</th></tr></thead>
            <tbody>
              {filtered.map((record) => <tr key={record.agent_id} aria-selected={record.agent_id === selectedRecord?.agent_id} onClick={() => setSelectedAgentId(record.agent_id)}><td><button className="table-select" type="button" onClick={(event) => { event.stopPropagation(); setSelectedAgentId(record.agent_id); }}>{record.agent_id}</button></td><td><span className={`text-status status-${stateClass(record.status)}`}>{record.status}</span></td><td className="mono">{record.parent_a ?? '—'}</td><td className="mono">{record.parent_b ?? '—'}</td><td>{record.reproduction_method}</td><td>{record.mutation_applied === null ? 'Unknown (legacy)' : record.mutation_applied ? 'Applied' : '—'}</td><td className="mono">{record.offspring_ids.length}</td></tr>)}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}