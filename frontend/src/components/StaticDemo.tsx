import { useState, useEffect } from 'react';
import type { InferenceResponse, LogitLensTrajectory } from '../types';
import LogitLensView from './LogitLensView';
import AttentionMap from './AttentionMap';
import TokenDisplay from './TokenDisplay';

interface ExportedScenario {
  name: string;
  model: string;
  model_info: {
    model_name: string;
    num_layers: number;
    num_heads: number;
    hidden_size: number;
    vocab_size: number;
  };
  prompt: string;
  input_tokens: string[];
  generated_tokens: string[];
  top_predictions: Array<Array<{
    token: string;
    probability: number;
    logit: number;
  }>> | null;
  attention_maps: Array<{
    layer_idx: number;
    weights: number[][][];
  }> | null;
  logit_lens_trajectory: LogitLensTrajectory;
  logit_lens_results: Record<string, {
    layer_idx: number;
    token_idx: number;
    top_tokens: string[];
    top_probs: number[];
    top_logits: number[];
    entropy: number;
  }> | null;
}

interface StaticDemoProps {
  scenarioUrl?: string;
  scenarioData?: ExportedScenario;
}

export default function StaticDemo({ scenarioUrl, scenarioData: initialData }: StaticDemoProps) {
  const [scenario, setScenario] = useState<ExportedScenario | null>(initialData || null);
  const [loading, setLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [activeViz, setActiveViz] = useState<'logit_lens' | 'attention'>('logit_lens');

  useEffect(() => {
    if (initialData || !scenarioUrl) return;

    const loadScenario = async () => {
      try {
        const response = await fetch(scenarioUrl);
        if (!response.ok) throw new Error('Failed to load scenario');
        const data = await response.json();
        setScenario(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadScenario();
  }, [scenarioUrl, initialData]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        <div className="text-gray-400">Loading scenario...</div>
      </div>
    );
  }

  if (error || !scenario) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        <div className="text-red-400">{error || 'No scenario data'}</div>
      </div>
    );
  }

  // Convert to InferenceResponse format for components
  const inferenceData: InferenceResponse = {
    prompt: scenario.prompt,
    input_tokens: scenario.input_tokens,
    generated_tokens: scenario.generated_tokens,
    top_predictions: scenario.top_predictions,
    attention_maps: scenario.attention_maps,
    logit_lens_results: scenario.logit_lens_results
      ? Object.fromEntries(
          Object.entries(scenario.logit_lens_results).map(([key, value]) => [
            parseInt(key),
            value,
          ])
        )
      : null,
    hidden_state_norms: null,
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">LLM Interpretability Demo</h1>
            <p className="text-sm text-gray-400">{scenario.name}</p>
          </div>
          <div className="text-sm text-gray-400">
            Model: {scenario.model_info.model_name} |
            {scenario.model_info.num_layers} layers |
            {scenario.model_info.num_heads} heads
          </div>
        </div>
      </header>

      <div className="p-6 space-y-6">
        {/* Prompt */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-sm font-medium mb-2">Prompt</h2>
          <p className="font-mono text-cyan-300">{scenario.prompt}</p>
        </div>

        {/* Tokens */}
        <div className="bg-gray-800 rounded-lg p-4">
          <TokenDisplay
            tokens={scenario.input_tokens}
            generatedTokens={scenario.generated_tokens}
          />
        </div>

        {/* Visualization Tabs */}
        <div className="border-b border-gray-700">
          <div className="flex gap-1">
            {[
              { id: 'logit_lens', label: 'Logit Lens' },
              { id: 'attention', label: 'Attention Maps' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveViz(tab.id as typeof activeViz)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition ${
                  activeViz === tab.id
                    ? 'border-cyan-500 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Visualization */}
        <div>
          {activeViz === 'logit_lens' && inferenceData.logit_lens_results && (
            <LogitLensView results={inferenceData.logit_lens_results} />
          )}
          {activeViz === 'attention' && inferenceData.attention_maps && (
            <AttentionMap
              attentionData={inferenceData.attention_maps}
              tokens={inferenceData.input_tokens}
            />
          )}
        </div>

        {/* Trajectory Summary */}
        {scenario.logit_lens_trajectory && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium mb-4">Prediction Evolution</h3>
            <div className="flex flex-wrap gap-2">
              {scenario.logit_lens_trajectory.layers.map((layer, idx) => (
                <div
                  key={layer}
                  className="px-3 py-2 bg-gray-700 rounded text-sm"
                >
                  <div className="text-gray-400 text-xs">Layer {layer}</div>
                  <div className="font-mono text-cyan-300">
                    {scenario.logit_lens_trajectory.top_predictions[idx]}
                  </div>
                  <div className="text-gray-500 text-xs">
                    {(scenario.logit_lens_trajectory.confidences[idx] * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 pt-4">
          Generated with LLM Interpretability Workbench
        </div>
      </div>
    </div>
  );
}
