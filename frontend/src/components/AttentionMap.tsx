import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import type { AttentionData } from '../types';
import { useTheme } from '../contexts/ThemeContext';

interface AttentionMapProps {
  attentionData: AttentionData[];
  tokens: string[];
}

export default function AttentionMap({ attentionData, tokens }: AttentionMapProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState<number | 'average'>('average');
  const [zoom, setZoom] = useState(1);
  const [colorScheme, setColorScheme] = useState<'blues' | 'viridis' | 'magma'>('blues');
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Theme-aware colors for D3 and inline styles
  const chartColors = useMemo(() => ({
    cellStroke: isDark ? '#374151' : '#d1d5db',
    labelFill: isDark ? '#9ca3af' : '#6b7280',
    hoverStroke: '#06b6d4',
    // For inline style fallbacks
    bgPrimary: isDark ? '#111827' : '#f3f4f6',
    bgSecondary: isDark ? '#1f2937' : '#ffffff',
    bgTertiary: isDark ? '#374151' : '#e5e7eb',
    textPrimary: isDark ? '#f3f4f6' : '#111827',
    textSecondary: isDark ? '#9ca3af' : '#4b5563',
    borderColor: isDark ? '#374151' : '#d1d5db',
  }), [isDark]);

  const currentLayerData = attentionData[selectedLayer];
  const numHeads = currentLayerData?.weights?.length ?? 0;

  const attentionMatrix = useMemo(() => {
    if (!currentLayerData?.weights) return null;

    if (selectedHead === 'average') {
      // Average across all heads
      const numTokens = currentLayerData.weights[0].length;
      const averaged: number[][] = Array(numTokens)
        .fill(null)
        .map(() => Array(numTokens).fill(0));

      for (let h = 0; h < numHeads; h++) {
        for (let i = 0; i < numTokens; i++) {
          for (let j = 0; j < numTokens; j++) {
            averaged[i][j] += currentLayerData.weights[h][i][j] / numHeads;
          }
        }
      }
      return averaged;
    } else {
      return currentLayerData.weights[selectedHead];
    }
  }, [currentLayerData, selectedHead, numHeads]);

  // Get color interpolator based on scheme
  const getColorInterpolator = useCallback(() => {
    switch (colorScheme) {
      case 'viridis': return d3.interpolateViridis;
      case 'magma': return d3.interpolateMagma;
      default: return d3.interpolateBlues;
    }
  }, [colorScheme]);

  useEffect(() => {
    if (!svgRef.current || !attentionMatrix || tokens.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 80, right: 20, bottom: 20, left: 80 };
    const cellSize = Math.min(30, 400 / tokens.length);
    const width = tokens.length * cellSize;
    const height = tokens.length * cellSize;
    const totalWidth = width + margin.left + margin.right;
    const totalHeight = height + margin.top + margin.bottom;

    svg.attr('width', '100%')
       .attr('height', Math.max(400, totalHeight))
       .attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`);

    // Create a container group for zoom/pan
    const container = svg.append('g')
      .attr('class', 'zoom-container');

    const g = container.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const colorScale = d3.scaleSequential(getColorInterpolator())
      .domain([0, 1]);

    // Draw cells
    const cells = g.selectAll('.cell')
      .data(attentionMatrix.flatMap((row, i) =>
        row.map((value, j) => ({ row: i, col: j, value }))
      ))
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => d.col * cellSize)
      .attr('y', d => d.row * cellSize)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', chartColors.cellStroke)
      .attr('stroke-width', 0.5);

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'viz-tooltip')
      .style('opacity', 0);

    cells.on('mouseover', function(event, d) {
      d3.select(this).attr('stroke', chartColors.hoverStroke).attr('stroke-width', 2);
      tooltip.transition().duration(100).style('opacity', 1);
      tooltip.html(`
        <div class="font-mono">
          <div>From: "${tokens[d.row]}"</div>
          <div>To: "${tokens[d.col]}"</div>
          <div>Attention: ${d.value.toFixed(4)}</div>
        </div>
      `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke', chartColors.cellStroke).attr('stroke-width', 0.5);
      tooltip.transition().duration(200).style('opacity', 0);
    });

    // Y-axis labels (source tokens)
    g.selectAll('.y-label')
      .data(tokens)
      .enter()
      .append('text')
      .attr('class', 'y-label')
      .attr('x', -5)
      .attr('y', (_, i) => i * cellSize + cellSize / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', chartColors.labelFill)
      .attr('font-size', Math.min(11, cellSize - 2))
      .attr('font-family', 'monospace')
      .text(d => d.length > 8 ? d.slice(0, 8) + '...' : d);

    // X-axis labels (target tokens)
    g.selectAll('.x-label')
      .data(tokens)
      .enter()
      .append('text')
      .attr('class', 'x-label')
      .attr('x', (_, i) => i * cellSize + cellSize / 2)
      .attr('y', -5)
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .attr('fill', chartColors.labelFill)
      .attr('font-size', Math.min(11, cellSize - 2))
      .attr('font-family', 'monospace')
      .attr('transform', (_, i) => `rotate(-45, ${i * cellSize + cellSize / 2}, -5)`)
      .text(d => d.length > 8 ? d.slice(0, 8) + '...' : d);

    // Add zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);
    zoomBehaviorRef.current = zoomBehavior;

    // Cleanup tooltip on unmount
    return () => {
      tooltip.remove();
    };
  }, [attentionMatrix, tokens, getColorInterpolator, chartColors]);

  // Reset zoom
  const handleResetZoom = useCallback(() => {
    if (svgRef.current && zoomBehaviorRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  // Zoom in/out
  const handleZoom = useCallback((factor: number) => {
    if (svgRef.current && zoomBehaviorRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(200)
        .call(zoomBehaviorRef.current.scaleBy, factor);
    }
  }, []);

  if (!attentionData || attentionData.length === 0) {
    return (
      <div className="text-gray-500 dark:text-gray-400 text-center py-8">
        No attention data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div
        className="flex flex-wrap gap-4 items-center bg-gray-200 dark:bg-gray-800 rounded-lg p-4"
        style={{ backgroundColor: chartColors.bgTertiary }}
      >
        <div>
          <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1" style={{ color: chartColors.textSecondary }}>Layer</label>
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-1.5 text-sm"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary, borderColor: chartColors.borderColor }}
          >
            {attentionData.map((_, i) => (
              <option key={i} value={i}>Layer {i}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1" style={{ color: chartColors.textSecondary }}>Head</label>
          <select
            value={selectedHead}
            onChange={(e) => setSelectedHead(e.target.value === 'average' ? 'average' : Number(e.target.value))}
            className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-1.5 text-sm"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary, borderColor: chartColors.borderColor }}
          >
            <option value="average">Average</option>
            {Array.from({ length: numHeads }, (_, i) => (
              <option key={i} value={i}>Head {i}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1" style={{ color: chartColors.textSecondary }}>Colors</label>
          <select
            value={colorScheme}
            onChange={(e) => setColorScheme(e.target.value as typeof colorScheme)}
            className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-1.5 text-sm"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary, borderColor: chartColors.borderColor }}
          >
            <option value="blues">Blues</option>
            <option value="viridis">Viridis</option>
            <option value="magma">Magma</option>
          </select>
        </div>
        <div className="flex-1" />

        {/* Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleZoom(0.8)}
            className="p-1.5 bg-gray-300 dark:bg-gray-700 hover:bg-gray-400 dark:hover:bg-gray-600 rounded"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary }}
            title="Zoom out"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
          <span className="text-sm text-gray-600 dark:text-gray-400 w-12 text-center" style={{ color: chartColors.textSecondary }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => handleZoom(1.25)}
            className="p-1.5 bg-gray-300 dark:bg-gray-700 hover:bg-gray-400 dark:hover:bg-gray-600 rounded"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary }}
            title="Zoom in"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 bg-gray-300 dark:bg-gray-700 hover:bg-gray-400 dark:hover:bg-gray-600 rounded"
            style={{ backgroundColor: chartColors.bgSecondary, color: chartColors.textPrimary }}
            title="Reset zoom"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400" style={{ color: chartColors.textSecondary }}>
          {tokens.length} tokens | {numHeads} heads
        </div>
      </div>

      {/* Heatmap */}
      <div
        ref={containerRef}
        className="bg-gray-200 dark:bg-gray-800 rounded-lg p-4 overflow-hidden"
        style={{ backgroundColor: chartColors.bgTertiary }}
      >
        <h3 className="text-sm font-medium mb-4" style={{ color: chartColors.textPrimary }}>
          Attention Weights - Layer {selectedLayer}
          {selectedHead !== 'average' && ` Head ${selectedHead}`}
          {selectedHead === 'average' && ' (Averaged)'}
        </h3>
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-2" style={{ color: chartColors.textSecondary }}>
          Scroll to zoom, drag to pan
        </div>
        <div className="flex justify-center overflow-hidden" style={{ cursor: 'grab' }}>
          <svg ref={svgRef} style={{ minHeight: '400px' }} />
        </div>
      </div>

      {/* Color Legend */}
      <div className="bg-gray-200 dark:bg-gray-800 rounded-lg p-4" style={{ backgroundColor: chartColors.bgTertiary }}>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-400" style={{ color: chartColors.textSecondary }}>Attention Weight:</span>
          <div className="flex items-center gap-2">
            <span className="text-sm" style={{ color: chartColors.textPrimary }}>0</span>
            <div
              className="w-48 h-4 rounded"
              style={{
                background: colorScheme === 'blues'
                  ? 'linear-gradient(to right, #1e3a5f, #3b82f6, #93c5fd)'
                  : colorScheme === 'viridis'
                  ? 'linear-gradient(to right, #440154, #21918c, #fde725)'
                  : 'linear-gradient(to right, #000004, #b63679, #fcfdbf)',
              }}
            />
            <span className="text-sm" style={{ color: chartColors.textPrimary }}>1</span>
          </div>
        </div>
      </div>
    </div>
  );
}
