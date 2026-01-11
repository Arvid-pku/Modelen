import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useTheme } from '../contexts/ThemeContext';

interface ActivationTraceProps {
  norms: Record<number, number[]>;
  tokens: string[];
}

const COLORS = [
  '#06b6d4', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
];

export default function ActivationTrace({ norms, tokens }: ActivationTraceProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const [viewMode, setViewMode] = useState<'by_layer' | 'by_token'>('by_layer');
  const [selectedTokens, setSelectedTokens] = useState<number[]>([]);

  // Theme-aware colors
  const colors = {
    grid: isDark ? '#374151' : '#d1d5db',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1f2937' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
    cardBg: isDark ? 'bg-gray-800' : 'bg-white',
    tableBorder: isDark ? 'border-gray-700' : 'border-gray-200',
    tableRowHover: isDark ? 'hover:bg-gray-700/30' : 'hover:bg-gray-100',
    selectBg: isDark ? 'bg-gray-700' : 'bg-white',
    selectBorder: isDark ? 'border-gray-600' : 'border-gray-300',
  };

  const layers = useMemo(() => {
    return Object.keys(norms).map(Number).sort((a, b) => a - b);
  }, [norms]);

  // Data for "by layer" view - each line is a token
  const byLayerData = useMemo(() => {
    return layers.map((layer) => {
      const layerNorms = norms[layer];
      const entry: Record<string, number | string> = { layer: `L${layer}` };
      tokens.forEach((_token, idx) => {
        if (layerNorms && layerNorms[idx] !== undefined) {
          entry[`token_${idx}`] = layerNorms[idx];
        }
      });
      return entry;
    });
  }, [layers, norms, tokens]);

  // Data for "by token" view - each line is a layer
  const byTokenData = useMemo(() => {
    return tokens.map((token, tokenIdx) => {
      const entry: Record<string, number | string> = {
        token: token.length > 10 ? token.slice(0, 10) + '...' : token,
        tokenIdx
      };
      layers.forEach((layer) => {
        const layerNorms = norms[layer];
        if (layerNorms && layerNorms[tokenIdx] !== undefined) {
          entry[`layer_${layer}`] = layerNorms[tokenIdx];
        }
      });
      return entry;
    });
  }, [tokens, layers, norms]);

  const toggleToken = (idx: number) => {
    setSelectedTokens((prev) =>
      prev.includes(idx)
        ? prev.filter((i) => i !== idx)
        : [...prev, idx]
    );
  };

  const displayTokens = selectedTokens.length > 0
    ? selectedTokens
    : tokens.slice(0, Math.min(5, tokens.length)).map((_, i) => i);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className={`flex gap-4 items-center ${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <div>
          <label className="block text-sm text-gray-500 dark:text-gray-400 mb-1">View Mode</label>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as 'by_layer' | 'by_token')}
            className={`${colors.selectBg} border ${colors.selectBorder} rounded px-3 py-1.5 text-sm`}
          >
            <option value="by_layer">Norm by Layer</option>
            <option value="by_token">Norm by Token</option>
          </select>
        </div>
        <div className="flex-1" />
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {layers.length} layers | {tokens.length} tokens
        </div>
      </div>

      {/* Token Selector */}
      {viewMode === 'by_layer' && (
        <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
          <label className="block text-sm text-gray-500 dark:text-gray-400 mb-2">Select Tokens to Display</label>
          <div className="flex flex-wrap gap-2">
            {tokens.map((token, idx) => (
              <button
                key={idx}
                onClick={() => toggleToken(idx)}
                className={`px-2 py-1 text-xs rounded font-mono transition ${
                  displayTokens.includes(idx)
                    ? 'bg-cyan-600 text-white'
                    : isDark
                      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {idx}: {token.length > 8 ? token.slice(0, 8) + '...' : token}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <h3 className="text-sm font-medium mb-4">
          Hidden State Norms {viewMode === 'by_layer' ? 'Across Layers' : 'Across Tokens'}
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {viewMode === 'by_layer' ? (
              <LineChart data={byLayerData}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis
                  dataKey="layer"
                  stroke={colors.axis}
                  tick={{ fill: colors.axis, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.axis}
                  tick={{ fill: colors.axis, fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: colors.tooltipBg,
                    border: `1px solid ${colors.tooltipBorder}`,
                    color: isDark ? '#f3f4f6' : '#111827'
                  }}
                />
                <Legend />
                {displayTokens.map((tokenIdx, i) => (
                  <Line
                    key={tokenIdx}
                    type="monotone"
                    dataKey={`token_${tokenIdx}`}
                    name={`"${tokens[tokenIdx]}"`}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                  />
                ))}
              </LineChart>
            ) : (
              <LineChart data={byTokenData}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis
                  dataKey="token"
                  stroke={colors.axis}
                  tick={{ fill: colors.axis, fontSize: 10 }}
                  height={60}
                  interval={0}
                />
                <YAxis
                  stroke={colors.axis}
                  tick={{ fill: colors.axis, fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: colors.tooltipBg,
                    border: `1px solid ${colors.tooltipBorder}`,
                    color: isDark ? '#f3f4f6' : '#111827'
                  }}
                />
                <Legend />
                {layers.filter((_, i) => i % Math.ceil(layers.length / 5) === 0).map((layer, i) => (
                  <Line
                    key={layer}
                    type="monotone"
                    dataKey={`layer_${layer}`}
                    name={`Layer ${layer}`}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                  />
                ))}
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Statistics Table */}
      <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <h3 className="text-sm font-medium mb-4">Layer Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className={`border-b ${colors.tableBorder}`}>
                <th className="text-left py-2 px-3">Layer</th>
                <th className="text-right py-2 px-3">Min Norm</th>
                <th className="text-right py-2 px-3">Max Norm</th>
                <th className="text-right py-2 px-3">Mean Norm</th>
                <th className="text-right py-2 px-3">Std Dev</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((layer) => {
                const layerNorms = norms[layer] || [];
                const min = Math.min(...layerNorms);
                const max = Math.max(...layerNorms);
                const mean = layerNorms.reduce((a, b) => a + b, 0) / layerNorms.length;
                const std = Math.sqrt(
                  layerNorms.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / layerNorms.length
                );
                return (
                  <tr key={layer} className={`border-b ${colors.tableBorder}/50 ${colors.tableRowHover}`}>
                    <td className="py-2 px-3 font-mono">{layer}</td>
                    <td className="py-2 px-3 text-right font-mono text-cyan-600 dark:text-cyan-400">{min.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right font-mono text-red-600 dark:text-red-400">{max.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right font-mono">{mean.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right font-mono text-gray-500 dark:text-gray-400">{std.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
