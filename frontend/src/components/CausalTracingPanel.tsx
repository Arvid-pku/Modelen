import { useState, useCallback } from 'react';

interface CausalTracingPanelProps {
  numLayers: number;
  numHeads: number; // For future use with head-level patching
  numTokens: number;
  onRunTrace: (
    cleanPrompt: string,
    corruptedPrompt: string,
    patchLayer: number,
    patchComponent: 'mlp_output' | 'attn_output',
    patchTokenIdx: number
  ) => Promise<void>;
  isLoading: boolean;
}

interface TracingResult {
  layer: number;
  tokenIdx: number;
  component: 'mlp_output' | 'attn_output';
  effectSize: number;
}

export default function CausalTracingPanel({
  numLayers,
  numHeads: _numHeads, // Reserved for future head-level patching
  numTokens,
  onRunTrace,
  isLoading,
}: CausalTracingPanelProps) {
  const [cleanPrompt, setCleanPrompt] = useState('The Eiffel Tower is located in');
  const [corruptedPrompt, setCorruptedPrompt] = useState('The Colosseum is located in');
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<'mlp_output' | 'attn_output'>('attn_output');
  const [tracingMode, setTracingMode] = useState<'manual' | 'sweep'>('manual');
  const [tracingResults, setTracingResults] = useState<TracingResult[]>([]);
  const [sweepProgress, setSweepProgress] = useState<{ current: number; total: number } | null>(null);

  // Create heatmap data from results
  const heatmapData = tracingResults.reduce((acc, result) => {
    const key = `${result.layer}-${result.tokenIdx}-${result.component}`;
    acc[key] = result.effectSize;
    return acc;
  }, {} as Record<string, number>);

  const maxEffect = Math.max(...tracingResults.map(r => Math.abs(r.effectSize)), 0.01);

  // Get color for effect size
  const getEffectColor = (effect: number) => {
    const normalized = effect / maxEffect;
    if (normalized > 0) {
      return `rgba(34, 197, 94, ${Math.abs(normalized)})`; // Green for positive
    } else {
      return `rgba(239, 68, 68, ${Math.abs(normalized)})`; // Red for negative
    }
  };

  // Manual single patch run
  const handleManualRun = useCallback(async () => {
    if (selectedLayer === null || selectedToken === null) return;

    await onRunTrace(
      cleanPrompt,
      corruptedPrompt,
      selectedLayer,
      selectedComponent,
      selectedToken
    );
  }, [cleanPrompt, corruptedPrompt, selectedLayer, selectedComponent, selectedToken, onRunTrace]);

  // Sweep all layers for a specific token
  const handleSweep = useCallback(async () => {
    if (selectedToken === null) return;

    setTracingResults([]);
    const total = numLayers * 2; // Both components
    let current = 0;

    const results: TracingResult[] = [];

    for (let layer = 0; layer < numLayers; layer++) {
      for (const component of ['attn_output', 'mlp_output'] as const) {
        setSweepProgress({ current, total });
        current++;

        // Simulate effect size (in real implementation, this would call the backend)
        // For now, we'll add a placeholder that shows the UI working
        results.push({
          layer,
          tokenIdx: selectedToken,
          component,
          effectSize: Math.random() * 2 - 1, // Placeholder
        });

        // Small delay to show progress
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }

    setTracingResults(results);
    setSweepProgress(null);
  }, [selectedToken, numLayers]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Causal Tracing</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setTracingMode('manual')}
            className={`px-2 py-1 text-xs rounded ${
              tracingMode === 'manual' ? 'bg-cyan-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Manual
          </button>
          <button
            onClick={() => setTracingMode('sweep')}
            className={`px-2 py-1 text-xs rounded ${
              tracingMode === 'sweep' ? 'bg-cyan-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Sweep
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-500 dark:text-gray-400">
        Identify which model components cause the prediction to change between two prompts.
      </p>

      {/* Prompt inputs */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
            Clean Prompt (correct answer)
          </label>
          <textarea
            value={cleanPrompt}
            onChange={(e) => setCleanPrompt(e.target.value)}
            className="w-full h-16 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100 resize-none"
            placeholder="The Eiffel Tower is located in"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
            Corrupted Prompt (wrong answer)
          </label>
          <textarea
            value={corruptedPrompt}
            onChange={(e) => setCorruptedPrompt(e.target.value)}
            className="w-full h-16 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100 resize-none"
            placeholder="The Colosseum is located in"
          />
        </div>
      </div>

      {/* Token selection */}
      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
          Token Position to Patch
        </label>
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={0}
            max={numTokens - 1}
            value={selectedToken ?? ''}
            onChange={(e) => setSelectedToken(e.target.value ? Number(e.target.value) : null)}
            placeholder="Token index"
            className="w-24 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          />
          <span className="text-xs text-gray-500">
            (0 to {numTokens - 1})
          </span>
        </div>
      </div>

      {tracingMode === 'manual' && (
        <>
          {/* Layer and component selection */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Layer</label>
              <input
                type="number"
                min={0}
                max={numLayers - 1}
                value={selectedLayer ?? ''}
                onChange={(e) => setSelectedLayer(e.target.value ? Number(e.target.value) : null)}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                placeholder="Layer"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Component</label>
              <select
                value={selectedComponent}
                onChange={(e) => setSelectedComponent(e.target.value as 'mlp_output' | 'attn_output')}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
              >
                <option value="attn_output">Attention</option>
                <option value="mlp_output">MLP</option>
              </select>
            </div>
          </div>

          <button
            onClick={handleManualRun}
            disabled={isLoading || selectedLayer === null || selectedToken === null}
            className="w-full px-3 py-2 bg-cyan-700 hover:bg-cyan-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm"
          >
            {isLoading ? 'Running...' : 'Run Patch Experiment'}
          </button>
        </>
      )}

      {tracingMode === 'sweep' && (
        <>
          <button
            onClick={handleSweep}
            disabled={isLoading || selectedToken === null || sweepProgress !== null}
            className="w-full px-3 py-2 bg-purple-700 hover:bg-purple-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm"
          >
            {sweepProgress
              ? `Sweeping... ${sweepProgress.current}/${sweepProgress.total}`
              : 'Sweep All Layers'
            }
          </button>

          {/* Heatmap visualization */}
          {tracingResults.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-500 dark:text-gray-400">Effect Heatmap</span>
                <div className="flex items-center gap-2">
                  <span className="text-red-500 dark:text-red-400">-</span>
                  <div className="w-16 h-2 bg-gradient-to-r from-red-500 via-gray-400 dark:via-gray-600 to-green-500 rounded" />
                  <span className="text-green-500 dark:text-green-400">+</span>
                </div>
              </div>

              {/* Component labels */}
              <div className="flex">
                <div className="w-12" />
                <div className="flex-1 grid grid-cols-2 gap-1 text-xs text-center text-gray-500 dark:text-gray-400">
                  <span>Attn</span>
                  <span>MLP</span>
                </div>
              </div>

              {/* Layer rows */}
              <div className="max-h-64 overflow-y-auto">
                {Array.from({ length: numLayers }, (_, layer) => (
                  <div key={layer} className="flex items-center gap-1 mb-1">
                    <span className="w-12 text-xs text-gray-500 text-right pr-2">L{layer}</span>
                    <div className="flex-1 grid grid-cols-2 gap-1">
                      {(['attn_output', 'mlp_output'] as const).map((component) => {
                        const key = `${layer}-${selectedToken}-${component}`;
                        const effect = heatmapData[key] || 0;
                        return (
                          <div
                            key={component}
                            className="h-4 rounded cursor-pointer hover:ring-1 hover:ring-gray-900/50 dark:hover:ring-white/50"
                            style={{ backgroundColor: getEffectColor(effect) }}
                            title={`Layer ${layer} ${component}: ${effect.toFixed(4)}`}
                            onClick={() => {
                              setSelectedLayer(layer);
                              setSelectedComponent(component);
                              setTracingMode('manual');
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Instructions */}
      <div className="text-xs text-gray-500 space-y-1">
        <p><strong className="text-gray-700 dark:text-gray-300">How it works:</strong></p>
        <ol className="list-decimal list-inside space-y-1 pl-2">
          <li>Enter a clean prompt that gets the right answer</li>
          <li>Enter a corrupted prompt that gets a wrong answer</li>
          <li>Select which token position to investigate</li>
          <li>Run experiments to see which components restore the correct behavior</li>
        </ol>
      </div>
    </div>
  );
}
