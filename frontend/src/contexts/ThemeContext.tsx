import { createContext, useContext, useState, ReactNode, useCallback, useLayoutEffect } from 'react';

type Theme = 'dark' | 'light';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_STORAGE_KEY = 'llm-workbench-theme';

// Helper to apply theme class to document - more robust version
function applyThemeClass(theme: Theme) {
  if (typeof document === 'undefined') return;

  const root = document.documentElement;

  // Remove both classes first to ensure clean state
  root.classList.remove('dark', 'light');

  // Add the correct class
  root.classList.add(theme);

  // Also set a data attribute as backup for CSS selectors
  root.setAttribute('data-theme', theme);
}

// Helper to get initial theme
function getInitialTheme(): Theme {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  // Check localStorage first
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'light' || stored === 'dark') {
    return stored;
  }

  // Check system preference
  if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    return 'light';
  }

  return 'dark';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  // Use useLayoutEffect to apply theme synchronously before paint
  useLayoutEffect(() => {
    applyThemeClass(theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setThemeState(prev => {
      const newTheme = prev === 'dark' ? 'light' : 'dark';
      // Apply immediately for instant feedback
      applyThemeClass(newTheme);
      return newTheme;
    });
  }, []);

  const setTheme = useCallback((newTheme: Theme) => {
    applyThemeClass(newTheme);
    setThemeState(newTheme);
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
