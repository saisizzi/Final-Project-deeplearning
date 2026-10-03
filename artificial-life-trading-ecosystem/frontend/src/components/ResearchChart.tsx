import type { ReactNode } from 'react';

export interface ResearchChartSeries {
  name: string;
  values: number[];
  tone: 'best' | 'mean' | 'median' | 'manual' | 'hold' | 'selected';
}

interface ResearchChartProps {
  series: ResearchChartSeries[];
  xLabels?: string[];
  formatValue?: (value: number) => string;
  ariaLabel: string;
  children?: ReactNode;
}

const width = 880;
const height = 252;
const inset = { top: 14, right: 22, bottom: 34, left: 92 };

export function ResearchChart({ series, xLabels = [], formatValue = (value) => value.toFixed(0), ariaLabel, children }: ResearchChartProps) {
  const available = series.filter((item) => item.values.some(Number.isFinite));
  const allValues = available.flatMap((item) => item.values).filter(Number.isFinite);
  if (!allValues.length) {
    return <div className="empty-state">No recorded observations are available for this chart.</div>;
  }

  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const magnitude = Math.max(Math.abs(rawMin), Math.abs(rawMax));
  const paddingFloor = magnitude < 1 ? 0.001 : 1;
  const padding = rawMax === rawMin
    ? Math.max(magnitude * 0.01, paddingFloor)
    : Math.max((rawMax - rawMin) * 0.12, paddingFloor);
  const min = rawMin - padding;
  const max = rawMax + padding;
  const plotWidth = width - inset.left - inset.right;
  const plotHeight = height - inset.top - inset.bottom;
  const maxPoints = Math.max(...available.map((item) => item.values.length));
  const x = (index: number) => inset.left + (maxPoints <= 1 ? plotWidth / 2 : (index / (maxPoints - 1)) * plotWidth);
  const y = (value: number) => inset.top + ((max - value) / (max - min)) * plotHeight;
  const ticks = Array.from({ length: 4 }, (_, index) => max - ((max - min) * index) / 3);

  return (
    <figure className="research-chart" aria-label={ariaLabel}>
      <div className="chart-frame">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
          <title>{ariaLabel}</title>
          {ticks.map((value, index) => {
            const yValue = inset.top + (plotHeight * index) / 3;
            return (
              <g key={index}>
                <line className="chart-grid" x1={inset.left} x2={width - inset.right} y1={yValue} y2={yValue} />
                <text className="chart-axis-label" x={inset.left - 10} y={yValue + 3} textAnchor="end">{formatValue(value)}</text>
              </g>
            );
          })}
          <line className="chart-grid chart-axis" x1={inset.left} x2={inset.left} y1={inset.top} y2={height - inset.bottom} />
          {available.map((item) => {
            const segments: string[][] = [];
            item.values.forEach((value, index) => {
              if (!Number.isFinite(value)) {
                if (segments.at(-1)?.length) segments.push([]);
                return;
              }
              if (!segments.length) segments.push([]);
              segments.at(-1)?.push(`${x(index)},${y(value)}`);
            });
            return (
              <g key={item.name}>
                {segments.filter((segment) => segment.length > 1).map((segment, index) => <polyline key={`${item.name}-segment-${index}`} className={`chart-line chart-line-${item.tone}`} points={segment.join(' ')} />)}
                {item.values.map((value, index) => Number.isFinite(value) && (
                  <circle
                    key={`${item.name}-${index}`}
                    className={`chart-point chart-point-${item.tone}`}
                    cx={x(index)}
                    cy={y(value)}
                    r={index === item.values.length - 1 ? 3.5 : 2.3}
                  >
                    <title>{`${item.name}${xLabels[index] ? ` · ${xLabels[index]}` : ` · ${index + 1}`} · ${formatValue(value)}`}</title>
                  </circle>
                ))}
              </g>
            );
          })}
          {maxPoints > 0 && <>
            <text className="chart-axis-label" x={inset.left} y={height - 9} textAnchor="start">{xLabels[0] ?? 'Start'}</text>
            {maxPoints > 1 && <text className="chart-axis-label" x={width - inset.right} y={height - 9} textAnchor="end">{xLabels[maxPoints - 1] ?? `Point ${maxPoints}`}</text>}
          </>}
        </svg>
      </div>
      <figcaption className="chart-caption">
        <div className="chart-legend">
          {available.map((item) => <span key={item.name}><i className={`legend-line ${item.tone}`} />{item.name}</span>)}
        </div>
        {children}
      </figcaption>
    </figure>
  );
}