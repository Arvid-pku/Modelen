import { useMemo, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, Area, ComposedChart, ScatterChart, Scatter, Cell
} from 'recharts';
import { useTheme } from '../contexts/ThemeContext';

interface ActivationStatsProps {
  norms: Record<number, number[]>;
  tokens: string[];
}

interface LayerStats {
  layer: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  p25: number;
  p50: number;
  p75: number;
  skewness: number;
  kurtosis: number;
  cv: number; // coefficient of variation
  entropy: number;
  sparsity: number; // % of values below threshold
}

// Calculate skewness
function calculateSkewness(values: number[], mean: number, std: number): number {
  if (std === 0) return 0;
  const n = values.length;
  const m3 = values.reduce((acc, v) => acc + Math.pow((v - mean) / std, 3), 0) / n;
  return m3;
}

// Calculate kurtosis (excess kurtosis)
function calculateKurtosis(values: number[], mean: number, std: number): number {
  if (std === 0) return 0;
  const n = values.length;
  const m4 = values.reduce((acc, v) => acc + Math.pow((v - mean) / std, 4), 0) / n;
  return m4 - 3; // Excess kurtosis
}

// Calculate entropy of discretized distribution
function calculateEntropy(values: number[], bins: number = 20): number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return 0;

  const binWidth = (max - min) / bins;
  const counts = new Array(bins).fill(0);

  for (const v of values) {
    const binIdx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    counts[binIdx]++;
  }

  const probs = counts.map(c => c / values.length).filter(p => p > 0);
  return -probs.reduce((acc, p) => acc + p * Math.log2(p), 0);
}

// Calculate sparsity (% of values below threshold)
function calculateSparsity(values: number[], threshold: number = 0.1): number {
  const count = values.filter(v => Math.abs(v) < threshold).length;
  return (count / values.length) * 100;
}

export default function ActivationStats({ norms, tokens }: ActivationStatsProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [activeView, setActiveView] = useState<'overview' | 'distribution' | 'correlation' | 'anomalies'>('overview');

  // Theme-aware colors for charts
  const chartColors = {
    grid: isDark ? '#374151' : '#d1d5db',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1f2937' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
    tooltipText: isDark ? '#f3f4f6' : '#111827',
    cardBg: isDark ? 'bg-gray-800' : 'bg-white',
    cardBorder: isDark ? 'border-gray-700' : 'border-gray-200',
  };

  // Calculate statistics for each layer
  const layerStats = useMemo(() => {
    const stats: LayerStats[] = [];
    const layers = Object.keys(norms).map(Number).sort((a, b) => a - b);

    for (const layer of layers) {
      const values = norms[layer];
      if (!values || values.length === 0) continue;

      const sorted = [...values].sort((a, b) => a - b);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);

      stats.push({
        layer,
        mean,
        std,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        p25: sorted[Math.floor(sorted.length * 0.25)],
        p50: sorted[Math.floor(sorted.length * 0.5)],
        p75: sorted[Math.floor(sorted.length * 0.75)],
        skewness: calculateSkewness(values, mean, std),
        kurtosis: calculateKurtosis(values, mean, std),
        cv: mean !== 0 ? (std / Math.abs(mean)) * 100 : 0,
        entropy: calculateEntropy(values),
        sparsity: calculateSparsity(values, mean * 0.1), // 10% of mean as threshold
      });
    }

    return stats;
  }, [norms]);

  // Token-wise statistics
  const tokenStats = useMemo(() => {
    const layers = Object.keys(norms).map(Number).sort((a, b) => a - b);

    return tokens.map((token, tokenIdx) => {
      const normsAcrossLayers = layers.map(layer => norms[layer]?.[tokenIdx] ?? 0);
      const mean = normsAcrossLayers.reduce((a, b) => a + b, 0) / normsAcrossLayers.length;
      const max = Math.max(...normsAcrossLayers);
      const min = Math.min(...normsAcrossLayers);

      // Calculate growth rate (last layer vs first layer)
      const firstLayerNorm = normsAcrossLayers[0] || 1;
      const lastLayerNorm = normsAcrossLayers[normsAcrossLayers.length - 1] || 1;
      const growthRate = ((lastLayerNorm - firstLayerNorm) / firstLayerNorm) * 100;

      return {
        token: token.length > 10 ? token.slice(0, 10) + '...' : token,
        fullToken: token,
        tokenIdx,
        mean,
        max,
        min,
        range: max - min,
        growthRate,
      };
    });
  }, [norms, tokens]);

  // Growth analysis - how much do norms grow through layers
  const growthData = useMemo(() => {
    const layers = Object.keys(norms).map(Number).sort((a, b) => a - b);
    return layers.map((layer, idx) => {
      const currentMean = norms[layer]?.reduce((a, b) => a + b, 0) / (norms[layer]?.length || 1);
      const prevMean = idx > 0 ? norms[layers[idx - 1]]?.reduce((a, b) => a + b, 0) / (norms[layers[idx - 1]]?.length || 1) : currentMean;

      return {
        layer: `L${layer}`,
        mean: currentMean,
        growth: idx > 0 ? ((currentMean - prevMean) / prevMean * 100) : 0,
      };
    });
  }, [norms]);

  // Anomaly detection - find outliers
  const anomalies = useMemo(() => {
    const results: { layer: number; tokenIdx: number; token: string; value: number; zscore: number }[] = [];
    const layers = Object.keys(norms).map(Number).sort((a, b) => a - b);

    for (const layer of layers) {
      const values = norms[layer];
      if (!values) continue;

      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);

      values.forEach((value, tokenIdx) => {
        const zscore = std !== 0 ? (value - mean) / std : 0;
        if (Math.abs(zscore) > 2) { // 2 standard deviations
          results.push({
            layer,
            tokenIdx,
            token: tokens[tokenIdx] || `[${tokenIdx}]`,
            value,
            zscore,
          });
        }
      });
    }

    return results.sort((a, b) => Math.abs(b.zscore) - Math.abs(a.zscore)).slice(0, 20);
  }, [norms, tokens]);

  // Layer correlation data (simplified - correlating adjacent layers)
  const correlationData = useMemo(() => {
    const layers = Object.keys(norms).map(Number).sort((a, b) => a - b);
    const results: { layer: string; correlation: number }[] = [];

    for (let i = 1; i < layers.length; i++) {
      const prev = norms[layers[i - 1]];
      const curr = norms[layers[i]];
      if (!prev || !curr) continue;

      // Pearson correlation
      const n = Math.min(prev.length, curr.length);
      const meanPrev = prev.slice(0, n).reduce((a, b) => a + b, 0) / n;
      const meanCurr = curr.slice(0, n).reduce((a, b) => a + b, 0) / n;

      let num = 0, denPrev = 0, denCurr = 0;
      for (let j = 0; j < n; j++) {
        const dPrev = prev[j] - meanPrev;
        const dCurr = curr[j] - meanCurr;
        num += dPrev * dCurr;
        denPrev += dPrev * dPrev;
        denCurr += dCurr * dCurr;
      }

      const correlation = Math.sqrt(denPrev * denCurr) !== 0
        ? num / Math.sqrt(denPrev * denCurr)
        : 0;

      results.push({ layer: `L${layers[i-1]}-L${layers[i]}`, correlation });
    }

    return results;
  }, [norms]);

  // Distribution shape data
  const distributionData = useMemo(() => {
    return layerStats.map(stat => ({
      layer: `L${stat.layer}`,
      skewness: stat.skewness,
      kurtosis: stat.kurtosis,
      entropy: stat.entropy,
      cv: stat.cv,
    }));
  }, [layerStats]);

  if (!norms || Object.keys(norms).length === 0) {
    return (
      <div className="text-gray-500 dark:text-gray-400 text-center py-8">
        No hidden state data available. Enable "include_hidden_states" in the request.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* View Selector */}
      <div className="flex gap-2 flex-wrap">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'distribution', label: 'Distribution Shape' },
          { id: 'correlation', label: 'Layer Correlation' },
          { id: 'anomalies', label: 'Anomalies' },
        ].map(view => (
          <button
            key={view.id}
            onClick={() => setActiveView(view.id as typeof activeView)}
            className={`px-3 py-1.5 text-sm rounded transition ${
              activeView === view.id
                ? 'bg-cyan-600 text-white'
                : isDark
                  ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {view.label}
          </button>
        ))}
      </div>

      {activeView === 'overview' && (
        <>
          {/* Layer Statistics Summary */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-4">Layer-wise Activation Statistics</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={layerStats}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} label={{ value: 'Layer', position: 'bottom', offset: -5, fill: chartColors.axis }} />
                  <YAxis tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} label={{ value: 'Norm', angle: -90, position: 'insideLeft', fill: chartColors.axis }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number) => value.toFixed(4)}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="p75" stackId="1" stroke="none" fill="#06b6d4" fillOpacity={0.2} name="75th %ile" />
                  <Area type="monotone" dataKey="p25" stackId="2" stroke="none" fill="#06b6d4" fillOpacity={0.2} name="25th %ile" />
                  <Line type="monotone" dataKey="mean" stroke="#06b6d4" strokeWidth={2} name="Mean" dot={false} />
                  <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} strokeDasharray="5 5" name="Max" dot={false} />
                  <Line type="monotone" dataKey="min" stroke="#22c55e" strokeWidth={1} strokeDasharray="5 5" name="Min" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Growth Analysis */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-4">Activation Growth Through Layers</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={growthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 10, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} label={{ value: '% Change', angle: -90, position: 'insideLeft', fill: chartColors.axis }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number) => `${value.toFixed(2)}%`}
                  />
                  <Bar dataKey="growth" name="Growth %">
                    {growthData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.growth >= 0 ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Token Statistics */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-4">Token-wise Activation Range & Growth</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={tokenStats} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis type="number" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis dataKey="token" type="category" tick={{ fontSize: 10, fill: chartColors.axis }} stroke={chartColors.axis} width={80} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number, name: string) => [
                      name === 'growthRate' ? `${value.toFixed(1)}%` : value.toFixed(4),
                      name === 'growthRate' ? 'Growth Rate' : name
                    ]}
                    labelFormatter={(label) => {
                      const stat = tokenStats.find(s => s.token === label);
                      return `Token: "${stat?.fullToken}"`;
                    }}
                  />
                  <Legend />
                  <Bar dataKey="min" stackId="a" fill="#22c55e" name="Min" />
                  <Bar dataKey="range" stackId="a" fill="#06b6d4" name="Range" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {activeView === 'distribution' && (
        <>
          {/* Skewness and Kurtosis */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-2">Distribution Shape by Layer</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Skewness measures asymmetry (0 = symmetric). Kurtosis measures tail heaviness (0 = normal distribution).
            </p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 10, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number) => value.toFixed(3)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="skewness" stroke="#f59e0b" strokeWidth={2} name="Skewness" dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="kurtosis" stroke="#8b5cf6" strokeWidth={2} name="Kurtosis" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Entropy and CV */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-2">Entropy & Coefficient of Variation</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Entropy measures distribution spread. CV (Coefficient of Variation) measures relative variability.
            </p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 10, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis yAxisId="left" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number, name: string) => [
                      name === 'cv' ? `${value.toFixed(2)}%` : value.toFixed(3),
                      name
                    ]}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="entropy" fill="#06b6d4" name="Entropy (bits)" />
                  <Line yAxisId="right" type="monotone" dataKey="cv" stroke="#ef4444" strokeWidth={2} name="CV (%)" dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sparsity */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-2">Layer Sparsity</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Percentage of activations below 10% of layer mean. Higher sparsity may indicate more selective processing.
            </p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={layerStats}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 10, fill: chartColors.axis }} stroke={chartColors.axis} />
                  <YAxis tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} domain={[0, 100]} label={{ value: '%', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number) => `${value.toFixed(1)}%`}
                  />
                  <Bar dataKey="sparsity" fill="#10b981" name="Sparsity %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {activeView === 'correlation' && (
        <>
          {/* Layer Correlation */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-2">Adjacent Layer Correlation</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Pearson correlation between adjacent layers. High correlation suggests similar token representations; low correlation indicates transformation.
            </p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={correlationData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="layer" tick={{ fontSize: 9, fill: chartColors.axis }} stroke={chartColors.axis} angle={-45} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} domain={[-1, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                    formatter={(value: number) => value.toFixed(4)}
                  />
                  <Bar dataKey="correlation" name="Correlation">
                    {correlationData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.correlation > 0.8 ? '#22c55e' : entry.correlation > 0.5 ? '#f59e0b' : '#ef4444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Correlation Summary */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-4">Correlation Insights</h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {correlationData.filter(c => c.correlation > 0.8).length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">High (&gt;0.8)</div>
              </div>
              <div className="p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded">
                <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                  {correlationData.filter(c => c.correlation > 0.5 && c.correlation <= 0.8).length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Medium (0.5-0.8)</div>
              </div>
              <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {correlationData.filter(c => c.correlation <= 0.5).length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Low (&le;0.5)</div>
              </div>
            </div>
          </div>
        </>
      )}

      {activeView === 'anomalies' && (
        <>
          {/* Anomaly Scatter Plot */}
          <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
            <h3 className="text-sm font-medium mb-2">Activation Anomalies</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Values more than 2 standard deviations from layer mean. These may indicate important tokens or processing hotspots.
            </p>
            {anomalies.length > 0 ? (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                    <XAxis type="number" dataKey="layer" name="Layer" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                    <YAxis type="number" dataKey="zscore" name="Z-Score" tick={{ fontSize: 12, fill: chartColors.axis }} stroke={chartColors.axis} />
                    <Tooltip
                      contentStyle={{ backgroundColor: chartColors.tooltipBg, border: `1px solid ${chartColors.tooltipBorder}`, color: chartColors.tooltipText }}
                      formatter={(value: number, name: string) => [value.toFixed(3), name]}
                      labelFormatter={(_, payload) => {
                        const data = payload?.[0]?.payload;
                        return data ? `Token: "${data.token}" (idx: ${data.tokenIdx})` : '';
                      }}
                    />
                    <Scatter data={anomalies} fill="#ef4444">
                      {anomalies.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.zscore > 0 ? '#ef4444' : '#3b82f6'} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                No significant anomalies detected (all values within 2 standard deviations)
              </div>
            )}
          </div>

          {/* Anomaly Table */}
          {anomalies.length > 0 && (
            <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
              <h3 className="text-sm font-medium mb-4">Top Anomalies</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-300 dark:border-gray-700">
                      <th className="text-left py-2 px-2">Token</th>
                      <th className="text-right py-2 px-2">Layer</th>
                      <th className="text-right py-2 px-2">Value</th>
                      <th className="text-right py-2 px-2">Z-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomalies.slice(0, 10).map((anomaly, idx) => (
                      <tr key={idx} className="border-b border-gray-200 dark:border-gray-700/50">
                        <td className="py-1 px-2 font-mono text-xs">"{anomaly.token}"</td>
                        <td className="text-right py-1 px-2 font-mono">L{anomaly.layer}</td>
                        <td className="text-right py-1 px-2 font-mono">{anomaly.value.toFixed(2)}</td>
                        <td className={`text-right py-1 px-2 font-mono ${anomaly.zscore > 0 ? 'text-red-500' : 'text-blue-500'}`}>
                          {anomaly.zscore > 0 ? '+' : ''}{anomaly.zscore.toFixed(2)}σ
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Detailed Statistics Table (always visible) */}
      <div className={`${chartColors.cardBg} rounded-lg p-4 border ${chartColors.cardBorder}`}>
        <h3 className="text-sm font-medium mb-4">Detailed Layer Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-300 dark:border-gray-700">
                <th className="text-left py-2 px-2">Layer</th>
                <th className="text-right py-2 px-2">Mean</th>
                <th className="text-right py-2 px-2">Std</th>
                <th className="text-right py-2 px-2">Min</th>
                <th className="text-right py-2 px-2">Max</th>
                <th className="text-right py-2 px-2">CV%</th>
                <th className="text-right py-2 px-2">Skew</th>
                <th className="text-right py-2 px-2">Kurt</th>
              </tr>
            </thead>
            <tbody>
              {layerStats.map((stat) => (
                <tr key={stat.layer} className="border-b border-gray-200 dark:border-gray-700/50">
                  <td className="py-1 px-2 font-mono">L{stat.layer}</td>
                  <td className="text-right py-1 px-2 font-mono">{stat.mean.toFixed(2)}</td>
                  <td className="text-right py-1 px-2 font-mono">{stat.std.toFixed(2)}</td>
                  <td className="text-right py-1 px-2 font-mono text-green-600 dark:text-green-500">{stat.min.toFixed(2)}</td>
                  <td className="text-right py-1 px-2 font-mono text-red-600 dark:text-red-500">{stat.max.toFixed(2)}</td>
                  <td className="text-right py-1 px-2 font-mono">{stat.cv.toFixed(1)}</td>
                  <td className={`text-right py-1 px-2 font-mono ${Math.abs(stat.skewness) > 1 ? 'text-yellow-600 dark:text-yellow-500' : ''}`}>
                    {stat.skewness.toFixed(2)}
                  </td>
                  <td className={`text-right py-1 px-2 font-mono ${Math.abs(stat.kurtosis) > 2 ? 'text-purple-600 dark:text-purple-500' : ''}`}>
                    {stat.kurtosis.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
