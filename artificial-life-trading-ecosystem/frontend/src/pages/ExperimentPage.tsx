import { formatCurrency, formatNumber } from '../components/format';
import { useExperiment } from '../contexts/experiment-context';
import { useSimulation } from '../contexts/simulation-context';
import { getArtifactManifest } from '../data/repositories';

export function ExperimentPage() {
  const experiment = useExperiment();
  const { state, experiments, selectExperiment, busy, error } = useSimulation();
  const manifest = getArtifactManifest(state);
  const selectedModel = state?.experiment_id.split(':').at(-1) ?? experiment.forecasting_backbone;
  const config = state?.population_config;

  return (
    <>
      <section className="lead-section experiment-lead">
        <p className="section-kicker">Technical provenance</p>
        <div className="section-heading-row"><h2>Experiment record</h2><span className="status-label status-evaluated">{state?.data_split ?? 'RECORDED SNAPSHOT'}</span></div>
        <div className="field-row">
          <div><label htmlFor="experiment-model">Interactive forecasting experiment</label><select id="experiment-model" value={selectedModel} onChange={(event) => void selectExperiment(event.target.value as 'SimpleRNN' | 'LSTM' | 'GRU')} disabled={busy}>
            {experiments.map((item) => <option key={item.experiment_id} value={item.model_name}>{item.label}</option>)}
          </select></div>
          <p>Each model keeps its own development-only population state.</p>
        </div>
        {error && <p className="error-state" role="alert">{error}</p>}
        <dl className="summary-grid">
          <div><dt>Experiment ID</dt><dd>{experiment.experiment_id}</dd></div>
          <div><dt>Experiment name</dt><dd>{experiment.experiment_name}</dd></div>
          <div><dt>Ticker</dt><dd>{experiment.ticker}</dd></div>
          <div><dt>Source</dt><dd>Repository model artifacts and interactive API</dd></div>
          <div><dt>Date range</dt><dd>{experiment.start_date} — {experiment.end_date}</dd></div>
          <div><dt>Interval</dt><dd>{experiment.interval}</dd></div>
          <div><dt>Dataset hash</dt><dd>{experiment.dataset_hash ?? 'Not recorded'}</dd></div>
          <div><dt>Target mode</dt><dd>{experiment.target_mode}</dd></div>
          <div><dt>Features</dt><dd>{experiment.feature_set.join(', ')}</dd></div>
          <div><dt>Sequence length</dt><dd>{experiment.sequence_length}</dd></div>
          <div><dt>Selected interactive forecaster</dt><dd>{selectedModel}</dd></div>
          <div><dt>Random seed</dt><dd>{config?.seed ?? experiment.seed}</dd></div>
          <div><dt>Population size</dt><dd>{config?.population_size ?? experiment.population_size} organisms</dd></div>
          <div><dt>Generation length</dt><dd>{state?.generation_days ?? 10} trading days</dd></div>
          <div><dt>Generation count</dt><dd>{state?.generations.filter((generation) => generation.status === 'EVALUATED').length ?? experiment.generations} evaluated</dd></div>
          <div><dt>Starting capital</dt><dd>{formatCurrency(config?.starting_capital ?? experiment.starting_capital)}</dd></div>
          <div><dt>Profit requirement</dt><dd>Ending capital above starting capital</dd></div>
          <div><dt>Mutation rate</dt><dd>{formatNumber((config?.mutation_rate ?? experiment.mutation_rate) * 100, 1)}%</dd></div>
          <div><dt>Mutation sigma</dt><dd>{formatNumber(config?.mutation_sigma ?? experiment.mutation_sigma, 3)}</dd></div>
          <div><dt>Immigrant fraction</dt><dd>{formatNumber((config?.immigrant_fraction ?? experiment.immigrant_fraction) * 100, 1)}%</dd></div>
          <div><dt>Transaction cost</dt><dd>{formatNumber(experiment.transaction_cost * 100, 2)}%</dd></div>
        </dl>
      </section>

      <section>
        <p className="section-kicker">Artifact inventory</p><h2>Recorded measurements</h2>
        <ul className="plain-list">
          <li><span>Forecasting metrics</span><strong>{manifest.forecasting_metrics ? 'Recorded' : 'Not recorded'}</strong></li>
          <li><span>Generation metrics</span><strong>{manifest.generation_metrics ? 'Recorded' : 'Not recorded'}</strong></li>
          <li><span>Lineage data</span><strong>{manifest.lineage_data ? 'Recorded' : 'Not recorded'}</strong></li>
          <li><span>Strategy comparison</span><strong>{manifest.strategy_comparison ? 'Recorded' : 'Not recorded'}</strong></li>
          <li><span>Final-test results</span><strong>{manifest.final_test ? 'Recorded' : 'Not run'}</strong></li>
        </ul>
      </section>

      <section>
        <h2>Development status</h2>
        <p>{experiment.development_status}</p>
      </section>

      <section>
        <h2>Final test status</h2>
        <p>{experiment.final_test_status}</p>
      </section>
    </>
  );
}
