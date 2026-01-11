import { useState, useMemo } from 'react';
import type { InterventionConfig, ActivationPatch } from '../types';
import { useTheme } from '../contexts/ThemeContext';

interface InterventionPanelProps {
  numLayers: number;
  numHeads: number;
  interventions: InterventionConfig;
  onChange: (config: InterventionConfig) => void;
}

export default function InterventionPanel({
  numLayers,
  numHeads,
  interventions,
  onChange,
}: InterventionPanelProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [activeTab, setActiveTab] = useState<'skip' | 'ablate' | 'patch'>('skip');

  // Theme colors for inline styles
  const colors = useMemo(() => ({
    bgPrimary: isDark ? '#111827' : '#f3f4f6',
    bgSecondary: isDark ? '#1f2937' : '#ffffff',
    bgTertiary: isDark ? '#374151' : '#e5e7eb',
    textPrimary: isDark ? '#f3f4f6' : '#111827',
    textSecondary: isDark ? '#9ca3af' : '#4b5563',
    borderColor: isDark ? '#374151' : '#d1d5db',
  }), [isDark]);

  // Layer skipping
  const toggleLayerSkip = (layer: number) => {
    const newSkipLayers = interventions.skip_layers.includes(layer)
      ? interventions.skip_layers.filter((l) => l !== layer)
      : [...interventions.skip_layers, layer];
    onChange({ ...interventions, skip_layers: newSkipLayers });
  };

  // Head ablation
  const addHeadAblation = (layer: number, head: number) => {
    const exists = interventions.ablate_heads.some(
      (a) => a.layer === layer && a.head === head
    );
    if (!exists) {
      onChange({
        ...interventions,
        ablate_heads: [...interventions.ablate_heads, { layer, head }],
      });
    }
  };

  const removeHeadAblation = (layer: number, head: number) => {
    onChange({
      ...interventions,
      ablate_heads: interventions.ablate_heads.filter(
        (a) => !(a.layer === layer && a.head === head)
      ),
    });
  };

  // Activation patching
  const [patchForm, setPatchForm] = useState<Partial<ActivationPatch>>({
    layer: 0,
    component: 'mlp_output',
    token_index: 0,
    dim_index: 0,
    value: 0,
  });

  const addPatch = () => {
    if (
      patchForm.layer !== undefined &&
      patchForm.component &&
      patchForm.token_index !== undefined &&
      patchForm.dim_index !== undefined &&
      patchForm.value !== undefined
    ) {
      onChange({
        ...interventions,
        activation_patching: [
          ...interventions.activation_patching,
          patchForm as ActivationPatch,
        ],
      });
      setPatchForm({
        layer: 0,
        component: 'mlp_output',
        token_index: 0,
        dim_index: 0,
        value: 0,
      });
    }
  };

  const removePatch = (index: number) => {
    onChange({
      ...interventions,
      activation_patching: interventions.activation_patching.filter((_, i) => i !== index),
    });
  };

  const clearAll = () => {
    onChange({
      skip_layers: [],
      ablate_heads: [],
      activation_patching: [],
    });
  };

  const hasInterventions =
    interventions.skip_layers.length > 0 ||
    interventions.ablate_heads.length > 0 ||
    interventions.activation_patching.length > 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Interventions</h3>
        {hasInterventions && (
          <button
            onClick={clearAll}
            className="text-xs text-red-400 hover:text-red-300"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-200 dark:bg-gray-800 rounded p-1" style={{ backgroundColor: colors.bgTertiary }}>
        {[
          { id: 'skip', label: 'Layer Skip' },
          { id: 'ablate', label: 'Head Ablate' },
          { id: 'patch', label: 'Patch' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex-1 px-2 py-1.5 text-xs rounded transition ${
              activeTab === tab.id
                ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
            style={activeTab === tab.id
              ? { backgroundColor: colors.bgSecondary, color: colors.textPrimary }
              : { color: colors.textSecondary }
            }
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Layer Skipping */}
      {activeTab === 'skip' && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Select layers to skip (residual stream passes through unchanged)
          </p>
          <div className="grid grid-cols-6 gap-1">
            {Array.from({ length: numLayers }, (_, i) => (
              <button
                key={i}
                onClick={() => toggleLayerSkip(i)}
                className={`px-2 py-1 text-xs rounded font-mono transition ${
                  interventions.skip_layers.includes(i)
                    ? 'bg-red-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {i}
              </button>
            ))}
          </div>
          {interventions.skip_layers.length > 0 && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Skipping: {interventions.skip_layers.sort((a, b) => a - b).join(', ')}
            </p>
          )}
        </div>
      )}

      {/* Head Ablation */}
      {activeTab === 'ablate' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500 dark:text-gray-400" style={{ color: colors.textSecondary }}>
            Select attention heads to zero out
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Layer</label>
              <select
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
                onChange={() => {
                  document.getElementById('head-select')?.focus();
                }}
                id="layer-select"
              >
                {Array.from({ length: numLayers }, (_, i) => (
                  <option key={i} value={i}>Layer {i}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Head</label>
              <select
                id="head-select"
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
                onChange={(e) => {
                  const layerSelect = document.getElementById('layer-select') as HTMLSelectElement;
                  const layer = Number(layerSelect?.value || 0);
                  const head = Number(e.target.value);
                  addHeadAblation(layer, head);
                }}
              >
                <option value="">Select head...</option>
                {Array.from({ length: numHeads }, (_, i) => (
                  <option key={i} value={i}>Head {i}</option>
                ))}
              </select>
            </div>
          </div>
          {interventions.ablate_heads.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-gray-500 dark:text-gray-400">Active ablations:</p>
              <div className="flex flex-wrap gap-1">
                {interventions.ablate_heads.map((ablation, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded text-xs"
                  >
                    L{ablation.layer}H{ablation.head}
                    <button
                      onClick={() => removeHeadAblation(ablation.layer, ablation.head)}
                      className="hover:text-red-500 dark:hover:text-red-100"
                    >
                      x
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Activation Patching */}
      {activeTab === 'patch' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500 dark:text-gray-400" style={{ color: colors.textSecondary }}>
            Overwrite specific activation values
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Layer</label>
              <input
                type="number"
                min={0}
                max={numLayers - 1}
                value={patchForm.layer}
                onChange={(e) => setPatchForm({ ...patchForm, layer: Number(e.target.value) })}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Component</label>
              <select
                value={patchForm.component}
                onChange={(e) => setPatchForm({ ...patchForm, component: e.target.value as 'mlp_output' | 'attn_output' })}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
              >
                <option value="mlp_output">MLP Output</option>
                <option value="attn_output">Attention Output</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Token Index</label>
              <input
                type="number"
                min={0}
                value={patchForm.token_index}
                onChange={(e) => setPatchForm({ ...patchForm, token_index: Number(e.target.value) })}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Dim Index</label>
              <input
                type="number"
                min={0}
                value={patchForm.dim_index}
                onChange={(e) => setPatchForm({ ...patchForm, dim_index: Number(e.target.value) })}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
              />
            </div>
            <div className="col-span-2">
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1" style={{ color: colors.textSecondary }}>Value</label>
              <input
                type="number"
                step="0.1"
                value={patchForm.value}
                onChange={(e) => setPatchForm({ ...patchForm, value: Number(e.target.value) })}
                className="w-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary, borderColor: colors.borderColor }}
              />
            </div>
          </div>
          <button
            onClick={addPatch}
            className="w-full px-3 py-1.5 bg-cyan-700 hover:bg-cyan-600 text-white rounded text-sm"
          >
            Add Patch
          </button>
          {interventions.activation_patching.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-gray-500 dark:text-gray-400">Active patches:</p>
              {interventions.activation_patching.map((patch, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 bg-gray-200 dark:bg-gray-700/50 rounded text-xs"
                >
                  <span className="font-mono text-gray-700 dark:text-gray-300">
                    L{patch.layer} {patch.component} [t{patch.token_index}, d{patch.dim_index}] = {patch.value}
                  </span>
                  <button
                    onClick={() => removePatch(idx)}
                    className="text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300"
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Summary */}
      {hasInterventions && (
        <div className="mt-4 p-2 bg-yellow-100 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700/50 rounded text-xs text-yellow-800 dark:text-yellow-200">
          Active interventions: {interventions.skip_layers.length} layer skips,{' '}
          {interventions.ablate_heads.length} head ablations,{' '}
          {interventions.activation_patching.length} patches
        </div>
      )}
    </div>
  );
}
