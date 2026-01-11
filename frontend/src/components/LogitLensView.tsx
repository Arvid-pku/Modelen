import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { LogitLensLayerResult } from '../types';
import { useTheme } from '../contexts/ThemeContext';

interface LogitLensViewProps {
  results: Record<number, LogitLensLayerResult>;
}

export default function LogitLensView({ results }: LogitLensViewProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  // Theme-aware colors
  const colors = {
    grid: isDark ? '#374151' : '#d1d5db',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1f2937' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
    cardBg: isDark ? 'bg-gray-800' : 'bg-white',
    tableBorder: isDark ? 'border-gray-700' : 'border-gray-200',
    tableRowHover: isDark ? 'hover:bg-gray-700/30' : 'hover:bg-gray-100',
    tokenBg: isDark ? 'bg-cyan-900/50' : 'bg-cyan-100',
    tokenText: isDark ? 'text-cyan-300' : 'text-cyan-700',
    secondaryTokenBg: isDark ? 'bg-gray-700' : 'bg-gray-200',
    secondaryTokenText: isDark ? 'text-gray-300' : 'text-gray-700',
  };

  const layers = useMemo(() => {
    return Object.keys(results)
      .map(Number)
      .sort((a, b) => a - b);
  }, [results]);

  const chartData = useMemo(() => {
    return layers.map((layer) => {
      const result = results[layer];
      return {
        layer,
        confidence: result.top_probs[0] * 100,
        entropy: result.entropy,
        topToken: result.top_tokens[0],
      };
    });
  }, [layers, results]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 70) return isDark ? 'text-cyan-400' : 'text-cyan-600';
    if (confidence > 30) return isDark ? 'text-yellow-400' : 'text-yellow-600';
    return isDark ? 'text-red-400' : 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Confidence Chart */}
      <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <h3 className="text-sm font-medium mb-4">Prediction Confidence Across Layers</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
              <XAxis
                dataKey="layer"
                stroke={colors.axis}
                tick={{ fill: colors.axis, fontSize: 12 }}
                label={{ value: 'Layer', position: 'bottom', fill: colors.axis }}
              />
              <YAxis
                stroke={colors.axis}
                tick={{ fill: colors.axis, fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: 'Confidence %', angle: -90, position: 'insideLeft', fill: colors.axis }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: colors.tooltipBg,
                  border: `1px solid ${colors.tooltipBorder}`,
                  color: isDark ? '#f3f4f6' : '#111827'
                }}
                labelFormatter={(value) => `Layer ${value}`}
                formatter={(value: number, name: string) => [
                  `${value.toFixed(1)}%`,
                  name === 'confidence' ? 'Confidence' : name,
                ]}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#06b6d4"
                strokeWidth={2}
                dot={{ fill: '#06b6d4', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Entropy Chart */}
      <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <h3 className="text-sm font-medium mb-4">Prediction Entropy Across Layers</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
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
                labelFormatter={(value) => `Layer ${value}`}
                formatter={(value: number) => [value.toFixed(3), 'Entropy']}
              />
              <Line
                type="monotone"
                dataKey="entropy"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ fill: '#f59e0b', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Layer-by-Layer Table */}
      <div className={`${colors.cardBg} rounded-lg p-4 border ${colors.tableBorder}`}>
        <h3 className="text-sm font-medium mb-4">Top Predictions by Layer</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className={`border-b ${colors.tableBorder}`}>
                <th className="text-left py-2 px-3">Layer</th>
                <th className="text-left py-2 px-3">Top 1</th>
                <th className="text-left py-2 px-3">Top 2</th>
                <th className="text-left py-2 px-3">Top 3</th>
                <th className="text-right py-2 px-3">Confidence</th>
                <th className="text-right py-2 px-3">Entropy</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((layer) => {
                const result = results[layer];
                const confidence = result.top_probs[0] * 100;
                return (
                  <tr key={layer} className={`border-b ${colors.tableBorder}/50 ${colors.tableRowHover}`}>
                    <td className="py-2 px-3 font-mono">{layer}</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 ${colors.tokenBg} rounded ${colors.tokenText} font-mono`}>
                        {result.top_tokens[0]}
                      </span>
                      <span className="ml-2 text-gray-500">
                        {(result.top_probs[0] * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 ${colors.secondaryTokenBg} rounded ${colors.secondaryTokenText} font-mono`}>
                        {result.top_tokens[1]}
                      </span>
                      <span className="ml-2 text-gray-500">
                        {(result.top_probs[1] * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 ${colors.secondaryTokenBg} rounded ${colors.secondaryTokenText} font-mono`}>
                        {result.top_tokens[2]}
                      </span>
                      <span className="ml-2 text-gray-500">
                        {(result.top_probs[2] * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className={`py-2 px-3 text-right font-mono ${getConfidenceColor(confidence)}`}>
                      {confidence.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-500 dark:text-gray-400">
                      {result.entropy.toFixed(3)}
                    </td>
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
