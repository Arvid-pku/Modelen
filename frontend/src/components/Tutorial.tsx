import { useState, useCallback } from 'react';

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for highlight
  action?: string; // Suggested action
}

const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to the LLM Interpretability Workbench',
    content: 'This tool helps you understand how large language models process text and make predictions. Let\'s take a quick tour of the main features.',
  },
  {
    id: 'prompt',
    title: 'Enter a Prompt',
    content: 'Start by entering a text prompt in the sidebar. The model will process this text and show you what\'s happening inside. Try a simple prompt like "The capital of France is".',
    target: 'textarea',
    action: 'Enter a prompt',
  },
  {
    id: 'templates',
    title: 'Use Prompt Templates',
    content: 'Click "Prompt Templates" to see pre-loaded examples designed to demonstrate different model behaviors like factual recall, pattern completion, and grammatical reasoning.',
  },
  {
    id: 'run',
    title: 'Run Analysis',
    content: 'Click "Run Analysis" (or press Ctrl+Enter) to process your prompt. The model will run inference and capture internal activations.',
    action: 'Click Run Analysis',
  },
  {
    id: 'logit-lens',
    title: 'Logit Lens View',
    content: 'The Logit Lens shows what the model would predict at each layer. Early layers often show basic patterns, while later layers refine toward the final prediction. Watch how the top predicted token changes through layers.',
  },
  {
    id: 'attention',
    title: 'Attention Maps',
    content: 'Attention maps show which tokens the model focuses on when processing each position. Bright cells indicate strong attention. Use the layer and head selectors to explore different attention patterns.',
  },
  {
    id: 'activation',
    title: 'Activation Trace',
    content: 'The activation trace shows the magnitude (norm) of hidden states at each layer and token position. This helps identify where the model is "active" and processing information.',
  },
  {
    id: 'statistics',
    title: 'Statistics View',
    content: 'The statistics tab provides numerical summaries of activation patterns - means, standard deviations, and percentiles across layers. This is useful for identifying anomalies or layer-specific behaviors.',
  },
  {
    id: 'interventions',
    title: 'Interventions',
    content: 'Use the Interventions panel to modify model behavior: skip layers to see their importance, ablate attention heads, or patch specific activations. Then compare results in Comparison mode.',
  },
  {
    id: 'tokens',
    title: 'Token Analysis',
    content: 'Click any token in the results to analyze that specific position. The visualizations will update to show what the model predicts at that position.',
  },
  {
    id: 'sessions',
    title: 'Save Your Work',
    content: 'Use the Sessions button to save your analysis results. You can export them as JSON files and reload them later - even without the model running.',
  },
  {
    id: 'share',
    title: 'Share Your Analysis',
    content: 'Click the share button to copy a link that includes your prompt and settings. Others can use this link to reproduce your analysis setup.',
  },
  {
    id: 'complete',
    title: 'You\'re Ready!',
    content: 'You now know the basics of the LLM Interpretability Workbench. Explore different prompts, try interventions, and discover how language models work internally. Press T to toggle dark/light theme, or ? to see all keyboard shortcuts.',
  },
];

interface TutorialProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Tutorial({ isOpen, onClose }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);

  const handleNext = useCallback(() => {
    if (currentStep < TUTORIAL_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
      setCurrentStep(0);
    }
  }, [currentStep, onClose]);

  const handlePrev = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const handleSkip = useCallback(() => {
    onClose();
    setCurrentStep(0);
  }, [onClose]);

  if (!isOpen) return null;

  const step = TUTORIAL_STEPS[currentStep];
  const isLastStep = currentStep === TUTORIAL_STEPS.length - 1;
  const isFirstStep = currentStep === 0;

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={handleSkip}
      />

      {/* Tutorial Card */}
      <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-lg">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-cyan-600 text-white px-6 py-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">{step.title}</h2>
              <button
                onClick={handleSkip}
                className="text-white/80 hover:text-white"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="text-sm text-white/80 mt-1">
              Step {currentStep + 1} of {TUTORIAL_STEPS.length}
            </div>
          </div>

          {/* Content */}
          <div className="px-6 py-4">
            <p className="text-gray-700 dark:text-gray-300">
              {step.content}
            </p>
            {step.action && (
              <div className="mt-3 p-2 bg-cyan-50 dark:bg-cyan-900/30 rounded text-sm text-cyan-700 dark:text-cyan-300">
                <strong>Try it:</strong> {step.action}
              </div>
            )}
          </div>

          {/* Progress Bar */}
          <div className="px-6">
            <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full">
              <div
                className="h-1 bg-cyan-500 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / TUTORIAL_STEPS.length) * 100}%` }}
              />
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 flex justify-between items-center">
            <button
              onClick={handleSkip}
              className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              Skip tutorial
            </button>
            <div className="flex gap-2">
              {!isFirstStep && (
                <button
                  onClick={handlePrev}
                  className="px-4 py-2 text-sm bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded"
                >
                  Previous
                </button>
              )}
              <button
                onClick={handleNext}
                className="px-4 py-2 text-sm bg-cyan-600 hover:bg-cyan-700 text-white rounded"
              >
                {isLastStep ? 'Finish' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// Hook to manage tutorial state
export function useTutorial() {
  const [isOpen, setIsOpen] = useState(() => {
    // Check if user has seen the tutorial
    return localStorage.getItem('llm-workbench-tutorial-seen') !== 'true';
  });

  const closeTutorial = useCallback(() => {
    setIsOpen(false);
    localStorage.setItem('llm-workbench-tutorial-seen', 'true');
  }, []);

  const openTutorial = useCallback(() => {
    setIsOpen(true);
  }, []);

  const resetTutorial = useCallback(() => {
    localStorage.removeItem('llm-workbench-tutorial-seen');
    setIsOpen(true);
  }, []);

  return {
    isOpen,
    closeTutorial,
    openTutorial,
    resetTutorial,
  };
}
