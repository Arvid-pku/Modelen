import { useCallback, useEffect } from 'react';
import type { InterventionConfig } from '../types';

interface ShareableState {
  prompt: string;
  viewMode: 'single' | 'comparison';
  activeViz: string;
  interventions?: InterventionConfig;
  selectedTokenIdx?: number;
  maxNewTokens?: number;
}

// Encode state to URL-safe base64
function encodeState(state: ShareableState): string {
  const json = JSON.stringify(state);
  // Use base64url encoding (URL-safe base64)
  return btoa(json)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

// Decode state from URL-safe base64
function decodeState(encoded: string): ShareableState | null {
  try {
    // Restore standard base64 padding
    let base64 = encoded
      .replace(/-/g, '+')
      .replace(/_/g, '/');
    while (base64.length % 4) {
      base64 += '=';
    }
    const json = atob(base64);
    return JSON.parse(json);
  } catch (e) {
    console.error('Failed to decode shared state:', e);
    return null;
  }
}

export function useShareableLink(
  currentState: ShareableState,
  onLoadState: (state: ShareableState) => void
) {
  // Load state from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sharedState = params.get('s');

    if (sharedState) {
      const decoded = decodeState(sharedState);
      if (decoded) {
        onLoadState(decoded);
        // Clear URL parameter after loading
        window.history.replaceState({}, '', window.location.pathname);
      }
    }
  }, []); // Only run on mount

  // Generate shareable link
  const generateLink = useCallback((): string => {
    const encoded = encodeState(currentState);
    const url = new URL(window.location.href);
    url.search = `?s=${encoded}`;
    return url.toString();
  }, [currentState]);

  // Copy link to clipboard
  const copyLink = useCallback(async (): Promise<boolean> => {
    const link = generateLink();
    try {
      await navigator.clipboard.writeText(link);
      return true;
    } catch (e) {
      console.error('Failed to copy link:', e);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = link;
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        return true;
      } catch {
        return false;
      } finally {
        document.body.removeChild(textArea);
      }
    }
  }, [generateLink]);

  return {
    generateLink,
    copyLink,
  };
}
