import { useMemo } from 'react';
import type { InferenceResponse } from '../types';
import AttentionMap from './AttentionMap';
import ActivationTrace from './ActivationTrace';

interface DiffViewProps {
  original: InferenceResponse;
  intervened: InferenceResponse;
  diff: {
    logit_diff: number[] | null;
    prediction_changed: boolean;
    original_prediction: string;
    intervened_prediction: string;
  };
  activeViz: 'logit_lens' | 'attention' | 'activation' | 'stats';
}

export default function DiffView({ original, intervened, diff, activeViz }: DiffViewProps) {
  // Calculate logit lens differences
  const logitLensDiff = useMemo(() => {
    if (!original.logit_lens_results || !intervened.logit_lens_results) return null;

    const diffResults: Record<number, {
      layer_idx: number;
      original_top: string;
      original_prob: number;
      intervened_top: string;
      intervened_prob: number;
      changed: boolean;
    }> = {};

    const layers = Object.keys(original.logit_lens_results).map(Number);
    for (const layer of layers) {
      const orig = original.logit_lens_results[layer];
      const interv = intervened.logit_lens_results?.[layer];
      if (orig && interv) {
        diffResults[layer] = {
          layer_idx: layer,
          original_top: orig.top_tokens[0],
          original_prob: orig.top_probs[0],
          intervened_top: interv.top_tokens[0],
          intervened_prob: interv.top_probs[0],
          changed: orig.top_tokens[0] !== interv.top_tokens[0],
        };
      }
    }
    return diffResults;
  }, [original, intervened]);

  return (
    <div className="space-y-6">
      {/* Prediction Change Banner */}
      <div className={`p-4 rounded-lg ${
        diff.prediction_changed
          ? 'bg-yellow-900/30 border border-yellow-700'
          : 'bg-green-900/30 border border-green-700'
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-medium">
              {diff.prediction_changed ? 'Prediction Changed' : 'Prediction Unchanged'}
            </span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="text-gray-400">Original: </span>
              <span className="font-mono px-2 py-0.5 bg-gray-700 rounded">
                {diff.original_prediction || '(empty)'}
              </span>
            </div>
            <span className="text-gray-500">&rarr;</span>
            <div>
              <span className="text-gray-400">Intervened: </span>
              <span className={`font-mono px-2 py-0.5 rounded ${
                diff.prediction_changed ? 'bg-yellow-700' : 'bg-gray-700'
              }`}>
                {diff.intervened_prediction || '(empty)'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Side-by-side or diff visualization based on active tab */}
      {activeViz === 'logit_lens' && logitLensDiff && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium mb-4">Logit Lens Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3">Layer</th>
                  <th className="text-left py-2 px-3">Original Top</th>
                  <th className="text-left py-2 px-3">Intervened Top</th>
                  <th className="text-right py-2 px-3">Prob Diff</th>
                  <th className="text-center py-2 px-3">Changed</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(logitLensDiff).map((row) => {
                  const probDiff = row.intervened_prob - row.original_prob;
                  return (
                    <tr
                      key={row.layer_idx}
                      className={`border-b border-gray-700/50 ${
                        row.changed ? 'bg-yellow-900/20' : ''
                      }`}
                    >
                      <td className="py-2 px-3 font-mono">{row.layer_idx}</td>
                      <td className="py-2 px-3">
                        <span className="px-2 py-0.5 bg-gray-700 rounded font-mono">
                          {row.original_top}
                        </span>
                        <span className="ml-2 text-gray-500">
                          {(row.original_prob * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-0.5 rounded font-mono ${
                          row.changed ? 'bg-yellow-700' : 'bg-gray-700'
                        }`}>
                          {row.intervened_top}
                        </span>
                        <span className="ml-2 text-gray-500">
                          {(row.intervened_prob * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className={`py-2 px-3 text-right font-mono ${
                        probDiff > 0 ? 'text-cyan-400' : probDiff < 0 ? 'text-red-400' : 'text-gray-500'
                      }`}>
                        {probDiff > 0 ? '+' : ''}{(probDiff * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-center">
                        {row.changed ? (
                          <span className="text-yellow-400">Yes</span>
                        ) : (
                          <span className="text-gray-500">-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeViz === 'attention' && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-400">Original</h3>
            {original.attention_maps && (
              <AttentionMap
                attentionData={original.attention_maps}
                tokens={original.input_tokens}
              />
            )}
          </div>
          <div>
            <h3 className="text-sm font-medium mb-2 text-yellow-400">Intervened</h3>
            {intervened.attention_maps && (
              <AttentionMap
                attentionData={intervened.attention_maps}
                tokens={intervened.input_tokens}
              />
            )}
          </div>
        </div>
      )}

      {activeViz === 'activation' && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-400">Original</h3>
            {original.hidden_state_norms && (
              <ActivationTrace
                norms={original.hidden_state_norms}
                tokens={original.input_tokens}
              />
            )}
          </div>
          <div>
            <h3 className="text-sm font-medium mb-2 text-yellow-400">Intervened</h3>
            {intervened.hidden_state_norms && (
              <ActivationTrace
                norms={intervened.hidden_state_norms}
                tokens={intervened.input_tokens}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
