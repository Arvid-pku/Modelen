import { useEffect, useCallback } from 'react';

interface ShortcutHandlers {
  onRun?: () => void;
  onSwitchTab?: (tabIndex: number) => void;
  onToggleTheme?: () => void;
  onSave?: () => void;
  onToggleSidebar?: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    const target = event.target as HTMLElement;
    const isInput = target.tagName === 'INPUT' ||
                    target.tagName === 'TEXTAREA' ||
                    target.isContentEditable;

    // Ctrl/Cmd + Enter to run (works even in inputs)
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      handlers.onRun?.();
      return;
    }

    // Ctrl/Cmd + S to save session
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault();
      handlers.onSave?.();
      return;
    }

    // Skip other shortcuts if in input
    if (isInput) return;

    // Number keys 1-4 to switch tabs
    if (event.key >= '1' && event.key <= '4') {
      event.preventDefault();
      handlers.onSwitchTab?.(parseInt(event.key) - 1);
      return;
    }

    // 'T' to toggle theme
    if (event.key === 't' || event.key === 'T') {
      event.preventDefault();
      handlers.onToggleTheme?.();
      return;
    }

    // 'B' to toggle sidebar (future)
    if (event.key === 'b' || event.key === 'B') {
      event.preventDefault();
      handlers.onToggleSidebar?.();
      return;
    }

    // '?' to show help (could be expanded later)
    if (event.key === '?') {
      event.preventDefault();
      // Could show a shortcuts modal
      console.log('Keyboard shortcuts:', {
        'Ctrl+Enter': 'Run analysis',
        'Ctrl+S': 'Save session',
        '1-3': 'Switch visualization tabs',
        'T': 'Toggle theme',
        'B': 'Toggle sidebar',
      });
      return;
    }
  }, [handlers]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

// Component to display keyboard shortcuts help
export const KEYBOARD_SHORTCUTS = [
  { keys: ['Ctrl', 'Enter'], description: 'Run analysis' },
  { keys: ['Ctrl', 'S'], description: 'Save session' },
  { keys: ['1'], description: 'Logit Lens tab' },
  { keys: ['2'], description: 'Attention Maps tab' },
  { keys: ['3'], description: 'Activation Trace tab' },
  { keys: ['4'], description: 'Statistics tab' },
  { keys: ['T'], description: 'Toggle theme' },
  { keys: ['?'], description: 'Show shortcuts' },
];
