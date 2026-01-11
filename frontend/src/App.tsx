import { useState, useCallback, useEffect, useRef } from 'react';
import { useModelInfo, useInference, useComparison, useLoadModel } from './hooks/useApi';
import { useKeyboardShortcuts, KEYBOARD_SHORTCUTS } from './hooks/useKeyboardShortcuts';
import { useShareableLink } from './hooks/useShareableLink';
import { useTheme } from './contexts/ThemeContext';
import type { InterventionConfig, InferenceResponse, ComparisonResponse } from './types';
import LogitLensView from './components/LogitLensView';
import AttentionMap from './components/AttentionMap';
import ActivationTrace from './components/ActivationTrace';
import InterventionPanel from './components/InterventionPanel';
import TokenDisplay from './components/TokenDisplay';
import DiffView from './components/DiffView';
import CausalTracingPanel from './components/CausalTracingPanel';
import ThemeToggle from './components/ThemeToggle';
import PromptTemplates from './components/PromptTemplates';
import ActivationStats from './components/ActivationStats';
import MemoryDisplay from './components/MemoryDisplay';
import Tutorial, { useTutorial } from './components/Tutorial';
import ExampleGallery from './components/ExampleGallery';

// Session type for save/load
interface SavedSession {
  id: string;
  name: string;
  timestamp: number;
  prompt: string;
  modelName: string;
  viewMode: 'single' | 'comparison';
  interventions: InterventionConfig;
  singleResult: InferenceResponse | null;
  comparisonResult: ComparisonResponse | null;
  selectedTokenIdx: number;
  maxNewTokens: number;
}

// Helper to generate unique ID
const generateId = () => Math.random().toString(36).substring(2, 15);

// Local storage key
const SESSIONS_STORAGE_KEY = 'llm-workbench-sessions';

// Available models organized by category
const AVAILABLE_MODELS = [
  { category: 'GPT-2', models: [
    { id: 'gpt2', name: 'GPT-2 (124M)' },
    { id: 'gpt2-medium', name: 'GPT-2 Medium (355M)' },
    { id: 'gpt2-large', name: 'GPT-2 Large (774M)' },
    { id: 'gpt2-xl', name: 'GPT-2 XL (1.5B)' },
  ]},
  { category: 'Qwen', models: [
    { id: 'Qwen/Qwen2-0.5B', name: 'Qwen2 0.5B' },
    { id: 'Qwen/Qwen2-1.5B', name: 'Qwen2 1.5B' },
    { id: 'Qwen/Qwen2.5-0.5B', name: 'Qwen2.5 0.5B' },
    { id: 'Qwen/Qwen2.5-1.5B', name: 'Qwen2.5 1.5B' },
  ]},
  { category: 'LLaMA', models: [
    { id: 'meta-llama/Llama-3.2-1B', name: 'LLaMA 3.2 1B' },
    { id: 'meta-llama/Llama-3.2-3B', name: 'LLaMA 3.2 3B' },
  ]},
  { category: 'Phi', models: [
    { id: 'microsoft/phi-1_5', name: 'Phi-1.5 (1.3B)' },
    { id: 'microsoft/phi-2', name: 'Phi-2 (2.7B)' },
  ]},
  { category: 'Gemma', models: [
    { id: 'google/gemma-2b', name: 'Gemma 2B' },
  ]},
  { category: 'Mistral', models: [
    { id: 'mistralai/Mistral-7B-v0.1', name: 'Mistral 7B' },
  ]},
  { category: 'OPT', models: [
    { id: 'facebook/opt-125m', name: 'OPT 125M' },
    { id: 'facebook/opt-350m', name: 'OPT 350M' },
    { id: 'facebook/opt-1.3b', name: 'OPT 1.3B' },
  ]},
  { category: 'BLOOM', models: [
    { id: 'bigscience/bloom-560m', name: 'BLOOM 560M' },
  ]},
  { category: 'GPT-Neo', models: [
    { id: 'EleutherAI/gpt-neo-125m', name: 'GPT-Neo 125M' },
    { id: 'EleutherAI/gpt-neo-1.3B', name: 'GPT-Neo 1.3B' },
  ]},
];

type ViewMode = 'single' | 'comparison';
type ActiveViz = 'logit_lens' | 'attention' | 'activation' | 'stats';

export default function App() {
  const [prompt, setPrompt] = useState('The capital of France is');
  const [viewMode, setViewMode] = useState<ViewMode>('single');
  const [activeViz, setActiveViz] = useState<ActiveViz>('logit_lens');
  const [interventions, setInterventions] = useState<InterventionConfig>({
    skip_layers: [],
    ablate_heads: [],
    activation_patching: [],
  });
  const [_selectedModel, setSelectedModel] = useState('gpt2');
  const [customModel, setCustomModel] = useState('');
  const [quantization, setQuantization] = useState<'4bit' | '8bit' | null>(null);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const modelSelectorRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modelSelectorRef.current && !modelSelectorRef.current.contains(event.target as Node)) {
        setShowModelSelector(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const [singleResult, setSingleResult] = useState<InferenceResponse | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(-1);
  const [maxNewTokens, setMaxNewTokens] = useState<number>(1);

  // Session management state
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>([]);
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const [sessionName, setSessionName] = useState('');
  const sessionPanelRef = useRef<HTMLDivElement>(null);

  // Load saved sessions from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(SESSIONS_STORAGE_KEY);
      if (stored) {
        setSavedSessions(JSON.parse(stored));
      }
    } catch (e) {
      console.error('Failed to load sessions:', e);
    }
  }, []);

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(savedSessions));
    } catch (e) {
      console.error('Failed to save sessions:', e);
    }
  }, [savedSessions]);

  // Close session panel when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (sessionPanelRef.current && !sessionPanelRef.current.contains(event.target as Node)) {
        setShowSessionPanel(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const { data: modelInfo, isLoading: modelLoading, refetch: refetchModelInfo } = useModelInfo();
  const inference = useInference();
  const comparison = useComparison();
  const loadModel = useLoadModel();
  const { toggleTheme } = useTheme();
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);

  // Shareable links
  const { copyLink } = useShareableLink(
    {
      prompt,
      viewMode,
      activeViz,
      interventions,
      selectedTokenIdx,
      maxNewTokens,
    },
    (state) => {
      setPrompt(state.prompt);
      setViewMode(state.viewMode);
      setActiveViz(state.activeViz as ActiveViz);
      if (state.interventions) setInterventions(state.interventions);
      if (state.selectedTokenIdx !== undefined) setSelectedTokenIdx(state.selectedTokenIdx);
      if (state.maxNewTokens !== undefined) setMaxNewTokens(state.maxNewTokens);
    }
  );

  const handleCopyLink = useCallback(async () => {
    const success = await copyLink();
    if (success) {
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 2000);
    }
  }, [copyLink]);

  // Tutorial
  const { isOpen: isTutorialOpen, closeTutorial, openTutorial } = useTutorial();

  // Example Gallery
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);

  const handleLoadModel = useCallback(async (modelId: string, useQuantization?: '4bit' | '8bit' | null) => {
    try {
      const quant = useQuantization !== undefined ? useQuantization : quantization;
      await loadModel.mutateAsync({ modelName: modelId, quantization: quant });
      setSelectedModel(modelId);
      setShowModelSelector(false);
      setSingleResult(null);
      setComparisonResult(null);
      // Refetch model info after loading
      setTimeout(() => refetchModelInfo(), 500);
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  }, [loadModel, refetchModelInfo, quantization]);

  const handleRun = useCallback(async () => {
    if (viewMode === 'single') {
      const hasInterventions =
        interventions.skip_layers.length > 0 ||
        interventions.ablate_heads.length > 0 ||
        interventions.activation_patching.length > 0;

      const result = await inference.mutateAsync({
        prompt,
        interventions: hasInterventions ? interventions : undefined,
        includeAttentions: true,
        includeLogitLens: true,
        includeHiddenStates: true,
        maxNewTokens,
        analyzeTokenIdx: selectedTokenIdx,
      });
      setSingleResult(result);
      setComparisonResult(null);
    } else {
      const result = await comparison.mutateAsync({
        prompt,
        interventions,
      });
      setComparisonResult(result);
      setSingleResult(null);
    }
  }, [prompt, viewMode, interventions, inference, comparison, maxNewTokens, selectedTokenIdx]);

  // Track if token was clicked (for auto-run)
  const tokenClickedRef = useRef(false);

  // Handler for clicking on a token to analyze it
  const handleTokenClick = useCallback((idx: number) => {
    setSelectedTokenIdx(idx);
    tokenClickedRef.current = true;
  }, []);

  // Auto-run analysis when token is clicked (if we already have results)
  useEffect(() => {
    if (tokenClickedRef.current && singleResult && !inference.isPending) {
      tokenClickedRef.current = false;
      // Trigger re-analysis for the new token
      handleRun();
    }
  }, [selectedTokenIdx, singleResult, inference.isPending, handleRun]);

  // Save current session
  const handleSaveSession = useCallback(() => {
    if (!singleResult && !comparisonResult) return;

    const session: SavedSession = {
      id: generateId(),
      name: sessionName || `Session ${new Date().toLocaleString()}`,
      timestamp: Date.now(),
      prompt,
      modelName: modelInfo?.model_name || 'unknown',
      viewMode,
      interventions,
      singleResult,
      comparisonResult,
      selectedTokenIdx,
      maxNewTokens,
    };

    setSavedSessions(prev => [session, ...prev]);
    setSessionName('');
  }, [singleResult, comparisonResult, sessionName, prompt, modelInfo, viewMode, interventions, selectedTokenIdx, maxNewTokens]);

  // Load a saved session
  const handleLoadSession = useCallback((session: SavedSession) => {
    setPrompt(session.prompt);
    setViewMode(session.viewMode);
    setInterventions(session.interventions);
    setSingleResult(session.singleResult);
    setComparisonResult(session.comparisonResult);
    setSelectedTokenIdx(session.selectedTokenIdx);
    setMaxNewTokens(session.maxNewTokens);
    setShowSessionPanel(false);
  }, []);

  // Delete a saved session
  const handleDeleteSession = useCallback((sessionId: string) => {
    setSavedSessions(prev => prev.filter(s => s.id !== sessionId));
  }, []);

  // Export session to JSON file
  const handleExportSession = useCallback((session: SavedSession) => {
    const json = JSON.stringify(session, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${session.name.replace(/[^a-z0-9]/gi, '_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  // Export all sessions
  const handleExportAllSessions = useCallback(() => {
    const json = JSON.stringify(savedSessions, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llm-workbench-sessions-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [savedSessions]);

  // Import sessions from JSON file
  const handleImportSessions = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string);
        // Check if it's a single session or array of sessions
        if (Array.isArray(data)) {
          setSavedSessions(prev => [...data, ...prev]);
        } else if (data.id && data.timestamp) {
          setSavedSessions(prev => [data, ...prev]);
        }
      } catch (err) {
        console.error('Failed to import sessions:', err);
        alert('Failed to import sessions. Invalid JSON file.');
      }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset input
  }, []);

  // Causal tracing handler
  const handleCausalTrace = useCallback(async (
    _cleanPrompt: string, // Reserved for future cross-prompt patching
    corruptedPrompt: string,
    patchLayer: number,
    patchComponent: 'mlp_output' | 'attn_output',
    patchTokenIdx: number
  ) => {
    // For causal tracing, we run comparison with patch
    const patchIntervention: InterventionConfig = {
      skip_layers: [],
      ablate_heads: [],
      activation_patching: [{
        layer: patchLayer,
        component: patchComponent,
        token_index: patchTokenIdx,
        dim_index: -1, // -1 means patch entire position
        value: 0, // Placeholder - actual patching needs backend support
      }],
    };

    // Run comparison between clean and corrupted with patch
    const result = await comparison.mutateAsync({
      prompt: corruptedPrompt,
      interventions: patchIntervention,
    });
    setComparisonResult(result);
    setViewMode('comparison');
  }, [comparison]);

  const [showCausalTracing, setShowCausalTracing] = useState(false);

  // Tab names for keyboard shortcuts
  const vizTabs: ActiveViz[] = ['logit_lens', 'attention', 'activation', 'stats'];

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onRun: handleRun,
    onSwitchTab: (index) => {
      if (index >= 0 && index < vizTabs.length) {
        setActiveViz(vizTabs[index]);
      }
    },
    onToggleTheme: toggleTheme,
    onSave: handleSaveSession,
  });

  const isLoading = inference.isPending || comparison.isPending;
  const isModelLoading = loadModel.isPending;
  const currentResult = viewMode === 'single' ? singleResult : comparisonResult?.original;

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      {/* Header */}
      <header className="border-b border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-6 py-4" style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">LLM Interpretability Workbench</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {modelLoading || isModelLoading ? 'Loading model...' : modelInfo ? `Model: ${modelInfo.model_name}` : 'No model loaded'}
            </p>
          </div>
          <div className="flex items-center gap-4">
            {modelInfo && (
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {modelInfo.num_layers} layers | {modelInfo.num_heads} heads | {modelInfo.device}
              </div>
            )}

            {/* Memory Display */}
            <MemoryDisplay />

            {/* Keyboard Shortcuts Help */}
            <div className="relative">
              <button
                onClick={() => setShowShortcuts(!showShortcuts)}
                className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-gray-600 dark:text-gray-300"
                title="Keyboard shortcuts"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </button>
              {showShortcuts && (
                <div className="absolute right-0 top-full mt-2 w-64 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl z-50 p-4">
                  <h3 className="font-semibold mb-3">Keyboard Shortcuts</h3>
                  <div className="space-y-2 text-sm">
                    {KEYBOARD_SHORTCUTS.map((shortcut, idx) => (
                      <div key={idx} className="flex justify-between">
                        <span className="text-gray-500 dark:text-gray-400">{shortcut.description}</span>
                        <span className="flex gap-1">
                          {shortcut.keys.map((key, kidx) => (
                            <kbd
                              key={kidx}
                              className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs font-mono"
                            >
                              {key}
                            </kbd>
                          ))}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Tutorial/Help Button */}
            <button
              onClick={openTutorial}
              className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-gray-600 dark:text-gray-300"
              title="Open tutorial"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>

            {/* Share Button */}
            <button
              onClick={handleCopyLink}
              className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-gray-600 dark:text-gray-300"
              title="Copy shareable link"
            >
              {linkCopied ? (
                <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
              )}
            </button>

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Sessions Button */}
            <div className="relative" ref={sessionPanelRef}>
              <button
                onClick={() => setShowSessionPanel(!showSessionPanel)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 rounded text-sm flex items-center gap-2"
              >
                Sessions
                {savedSessions.length > 0 && (
                  <span className="bg-cyan-600 text-xs px-1.5 py-0.5 rounded-full">
                    {savedSessions.length}
                  </span>
                )}
              </button>

              {showSessionPanel && (
                <div className="absolute right-0 top-full mt-2 w-96 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl z-50 max-h-[32rem] overflow-hidden flex flex-col">
                  {/* Save current session */}
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-2">Save Current Analysis</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={sessionName}
                        onChange={(e) => setSessionName(e.target.value)}
                        placeholder="Session name (optional)"
                        className="flex-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                      />
                      <button
                        onClick={handleSaveSession}
                        disabled={!singleResult && !comparisonResult}
                        className="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:text-gray-200 dark:disabled:text-gray-400 text-white rounded text-sm"
                      >
                        Save
                      </button>
                    </div>
                  </div>

                  {/* Import/Export buttons */}
                  <div className="p-2 border-b border-gray-200 dark:border-gray-700 flex gap-2">
                    <label className="flex-1 px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 rounded text-sm text-center cursor-pointer">
                      Import
                      <input
                        type="file"
                        accept=".json"
                        onChange={handleImportSessions}
                        className="hidden"
                      />
                    </label>
                    <button
                      onClick={handleExportAllSessions}
                      disabled={savedSessions.length === 0}
                      className="flex-1 px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 disabled:opacity-50 rounded text-sm"
                    >
                      Export All
                    </button>
                  </div>

                  {/* Session list */}
                  <div className="overflow-y-auto flex-1">
                    {savedSessions.length === 0 ? (
                      <div className="p-4 text-center text-gray-500 text-sm">
                        No saved sessions
                      </div>
                    ) : (
                      savedSessions.map((session) => (
                        <div
                          key={session.id}
                          className="p-3 border-b border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-750"
                        >
                          <div className="flex items-start justify-between mb-1">
                            <div
                              className="flex-1 cursor-pointer"
                              onClick={() => handleLoadSession(session)}
                            >
                              <div className="font-medium text-sm text-gray-900 dark:text-gray-100">{session.name}</div>
                              <div className="text-xs text-gray-500 dark:text-gray-400">
                                {new Date(session.timestamp).toLocaleString()}
                              </div>
                              <div className="text-xs text-gray-400 dark:text-gray-500 truncate mt-1">
                                {session.prompt.slice(0, 50)}...
                              </div>
                              <div className="text-xs text-cyan-600 dark:text-cyan-500 mt-1">
                                Model: {session.modelName} | Mode: {session.viewMode}
                              </div>
                            </div>
                            <div className="flex gap-1 ml-2">
                              <button
                                onClick={() => handleExportSession(session)}
                                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                                title="Export"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                </svg>
                              </button>
                              <button
                                onClick={() => handleDeleteSession(session.id)}
                                className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400"
                                title="Delete"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </button>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="relative" ref={modelSelectorRef}>
              <button
                onClick={() => setShowModelSelector(!showModelSelector)}
                disabled={isModelLoading}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:bg-gray-300 dark:disabled:bg-gray-800 text-gray-700 dark:text-gray-200 rounded text-sm flex items-center gap-2"
              >
                {isModelLoading ? 'Loading...' : 'Change Model'}
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelSelector && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl z-50 max-h-[32rem] overflow-y-auto">
                  {/* Quantization selector */}
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-2">Quantization (GPU only)</label>
                    <div className="flex gap-2">
                      {[
                        { value: null, label: 'None' },
                        { value: '8bit' as const, label: '8-bit' },
                        { value: '4bit' as const, label: '4-bit' },
                      ].map((opt) => (
                        <button
                          key={opt.label}
                          onClick={() => setQuantization(opt.value)}
                          className={`flex-1 px-2 py-1 text-xs rounded ${
                            quantization === opt.value
                              ? 'bg-cyan-600 text-white'
                              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                          }`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {quantization ? `Load larger models with ${quantization} precision` : 'Full precision (more VRAM needed)'}
                    </p>
                  </div>

                  {/* Custom model input */}
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Custom Model (HuggingFace ID)</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={customModel}
                        onChange={(e) => setCustomModel(e.target.value)}
                        placeholder="e.g., microsoft/phi-2"
                        className="flex-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                      />
                      <button
                        onClick={() => customModel && handleLoadModel(customModel)}
                        disabled={!customModel || isModelLoading}
                        className="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-400 dark:disabled:bg-gray-600 text-white rounded text-sm"
                      >
                        Load
                      </button>
                    </div>
                  </div>

                  {/* Model categories */}
                  {AVAILABLE_MODELS.map((category) => (
                    <div key={category.category}>
                      <div className="px-3 py-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-750 sticky top-0">
                        {category.category}
                      </div>
                      {category.models.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => handleLoadModel(model.id)}
                          disabled={isModelLoading}
                          className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-900 dark:text-gray-100 ${
                            modelInfo?.model_name === model.id ? 'bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-300' : ''
                          }`}
                        >
                          <div>{model.name}</div>
                          <div className="text-xs text-gray-500 font-mono">{model.id}</div>
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Model loading error */}
        {loadModel.error && (
          <div className="mt-2 p-2 bg-red-900/50 border border-red-700 rounded text-sm text-red-200">
            Failed to load model: {loadModel.error.message}
          </div>
        )}
      </header>

      {/* Loading overlay */}
      {isModelLoading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 text-center">
            <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-900 dark:text-gray-100">Loading model...</p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">This may take a few minutes</p>
          </div>
        </div>
      )}

      <div className="flex h-[calc(100vh-73px)] bg-gray-100 dark:bg-gray-900" style={{ backgroundColor: 'var(--bg-primary)' }}>
        {/* Left Panel - Controls */}
        <aside className="w-80 border-r border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800/50 p-4 overflow-y-auto" style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          {/* Prompt Templates */}
          <PromptTemplates onSelect={setPrompt} />

          {/* Example Gallery Button */}
          <button
            onClick={() => setIsGalleryOpen(true)}
            className="w-full mb-4 flex items-center justify-center gap-2 px-3 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white rounded text-sm font-medium transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Example Gallery
          </button>

          {/* Prompt Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-24 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500 text-gray-900 dark:text-gray-100"
              style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
              placeholder="Enter your prompt..."
            />
          </div>

          {/* View Mode Toggle */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">View Mode</label>
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('single')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  viewMode === 'single'
                    ? 'bg-cyan-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                Single Run
              </button>
              <button
                onClick={() => setViewMode('comparison')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  viewMode === 'comparison'
                    ? 'bg-cyan-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                Compare
              </button>
            </div>
          </div>

          {/* Generation Settings */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">Generation Settings</label>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Max New Tokens</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={maxNewTokens}
                    onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="w-8 text-center text-sm">{maxNewTokens}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Generate and track multiple token predictions
                </p>
              </div>
              <div>
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Analyze Token Position</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="-1"
                    value={selectedTokenIdx}
                    onChange={(e) => setSelectedTokenIdx(parseInt(e.target.value))}
                    className="w-20 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
                    style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
                    placeholder="-1"
                  />
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {selectedTokenIdx === -1 ? '(last token)' : `(position ${selectedTokenIdx})`}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  -1 = last token, or click a token in results
                </p>
              </div>
            </div>
          </div>

          {/* Intervention Panel */}
          {modelInfo && (
            <InterventionPanel
              numLayers={modelInfo.num_layers}
              numHeads={modelInfo.num_heads}
              interventions={interventions}
              onChange={setInterventions}
            />
          )}

          {/* Causal Tracing Panel (collapsible) */}
          {modelInfo && (
            <div className="mt-4">
              <button
                onClick={() => setShowCausalTracing(!showCausalTracing)}
                className="w-full flex items-center justify-between px-3 py-2 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-750 rounded text-sm font-medium text-gray-700 dark:text-gray-200"
              >
                <span>Causal Tracing</span>
                <svg
                  className={`w-4 h-4 transition-transform ${showCausalTracing ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showCausalTracing && (
                <div className="mt-2 p-3 bg-gray-100 dark:bg-gray-800/50 rounded border border-gray-300 dark:border-gray-700">
                  <CausalTracingPanel
                    numLayers={modelInfo.num_layers}
                    numHeads={modelInfo.num_heads}
                    numTokens={singleResult?.input_tokens.length || 10}
                    onRunTrace={handleCausalTrace}
                    isLoading={isLoading}
                  />
                </div>
              )}
            </div>
          )}

          {/* Run Button */}
          <button
            onClick={handleRun}
            disabled={isLoading || !prompt}
            className="w-full mt-6 px-4 py-3 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded transition"
          >
            {isLoading ? 'Running...' : 'Run Analysis'}
          </button>

          {/* Error Display */}
          {(inference.error || comparison.error) && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded text-sm text-red-200">
              {inference.error?.message || comparison.error?.message}
            </div>
          )}
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-hidden flex flex-col">
          {/* Visualization Tabs */}
          <div className="border-b border-gray-300 dark:border-gray-700 px-4">
            <div className="flex gap-1">
              {[
                { id: 'logit_lens', label: 'Logit Lens' },
                { id: 'attention', label: 'Attention Maps' },
                { id: 'activation', label: 'Activation Trace' },
                { id: 'stats', label: 'Statistics' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveViz(tab.id as ActiveViz)}
                  className={`px-4 py-3 text-sm font-medium border-b-2 transition ${
                    activeViz === tab.id
                      ? 'border-cyan-500 text-cyan-600 dark:text-cyan-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Token Display */}
          {currentResult && (
            <div className="border-b border-gray-200 dark:border-gray-700 p-4">
              <TokenDisplay
                tokens={currentResult.input_tokens}
                generatedTokens={currentResult.generated_tokens}
                highlightIndex={selectedTokenIdx >= 0 ? selectedTokenIdx : undefined}
                onTokenClick={handleTokenClick}
              />
              {selectedTokenIdx >= 0 && (
                <div className="mt-2 flex items-center gap-2 text-sm">
                  <span className="text-gray-500 dark:text-gray-400">
                    Analyzing token {selectedTokenIdx}: "{currentResult.input_tokens[selectedTokenIdx]}"
                  </span>
                  <button
                    onClick={() => setSelectedTokenIdx(-1)}
                    className="text-cyan-600 dark:text-cyan-400 hover:text-cyan-500 dark:hover:text-cyan-300"
                  >
                    (reset to last)
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Visualization Area */}
          <div className="flex-1 overflow-auto p-4 bg-gray-100 dark:bg-gray-900" style={{ backgroundColor: 'var(--bg-primary)' }}>
            {!currentResult && !comparisonResult ? (
              <div className="h-full flex items-center justify-center text-gray-500">
                Run analysis to see visualizations
              </div>
            ) : viewMode === 'comparison' && comparisonResult ? (
              <DiffView
                original={comparisonResult.original}
                intervened={comparisonResult.intervened}
                diff={comparisonResult.diff}
                activeViz={activeViz}
              />
            ) : currentResult ? (
              <>
                {activeViz === 'logit_lens' && currentResult.logit_lens_results && (
                  <LogitLensView results={currentResult.logit_lens_results} />
                )}
                {activeViz === 'attention' && currentResult.attention_maps && (
                  <AttentionMap
                    attentionData={currentResult.attention_maps}
                    tokens={currentResult.input_tokens}
                  />
                )}
                {activeViz === 'activation' && currentResult.hidden_state_norms && (
                  <ActivationTrace
                    norms={currentResult.hidden_state_norms}
                    tokens={currentResult.input_tokens}
                  />
                )}
                {activeViz === 'stats' && currentResult.hidden_state_norms && (
                  <ActivationStats
                    norms={currentResult.hidden_state_norms}
                    tokens={currentResult.input_tokens}
                  />
                )}
              </>
            ) : null}
          </div>
        </main>
      </div>

      {/* Tutorial Modal */}
      <Tutorial isOpen={isTutorialOpen} onClose={closeTutorial} />

      {/* Example Gallery Modal */}
      <ExampleGallery
        isOpen={isGalleryOpen}
        onClose={() => setIsGalleryOpen(false)}
        onSelectExample={setPrompt}
      />
    </div>
  );
}
