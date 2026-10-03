import { NavLink, Outlet } from 'react-router-dom';

import { useExperiment } from '../contexts/experiment-context';
import { useSimulation } from '../contexts/simulation-context';

const navigation = [
  { label: 'Overview', to: '/' },
  { label: 'Forecasting', to: '/forecasting' },
  { label: 'Population', to: '/population' },
  { label: 'Lineage', to: '/lineage' },
  { label: 'Agents', to: '/agents' },
  { label: 'Backtest', to: '/backtest' },
  { label: 'Replay', to: '/replay' },
  { label: 'Experiment', to: '/experiment' },
];

export function AppLayout() {
  const experiment = useExperiment();
  const { state, error } = useSimulation();

  return (
    <div className="app-shell">
      <aside className="side-rail" aria-label="Research workspace">
          <NavLink className="brand-block" to="/" aria-label="Artificial-Life Trading Ecosystem overview">
          <span className="brand-mark" aria-hidden="true">AL</span>
          <span className="brand-copy">
            <strong>Artificial-Life</strong>
            <span>Trading ecosystem</span>
          </span>
          </NavLink>
        <p className="rail-heading">Workspace</p>
        <nav className="main-nav" aria-label="Main navigation">
          {navigation.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="rail-experiment">
          <span className="field-label">Selected experiment</span>
          <strong>{state?.experiment_id ?? experiment.experiment_id}</strong>
          <span className="mono quiet">{experiment.ticker} · {state?.data_split ?? 'development'}</span>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-title">
            <p className="eyebrow">Research workspace</p>
            <h1>{state ? `Generation ${state.generation}` : experiment.experiment_name}</h1>
          </div>
          <div className="topbar-state" aria-label="Current simulation state">
            {state && <>
              <span className={`status-label status-${state.status.toLowerCase()}`}>{state.status}</span>
              <span className="mono">{state.day} / {state.generation_days} days</span>
              {state.date && <time className="mono quiet">{state.date.slice(0, 10)}</time>}
            </>}
          </div>
        </header>

        <main className="page-shell">
          {error && <p className="error-state" role="alert">{error}</p>}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
