interface TokenDisplayProps {
  tokens: string[];
  generatedTokens: string[];
  highlightIndex?: number;
  onTokenClick?: (index: number) => void;
}

export default function TokenDisplay({
  tokens,
  generatedTokens,
  highlightIndex,
  onTokenClick,
}: TokenDisplayProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-sm text-gray-400">
        <span>Input Tokens ({tokens.length})</span>
        {generatedTokens.length > 0 && (
          <span>Generated ({generatedTokens.length})</span>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {tokens.map((token, idx) => (
          <span
            key={`input-${idx}`}
            onClick={() => onTokenClick?.(idx)}
            className={`
              px-2 py-1 rounded text-sm font-mono cursor-pointer transition
              ${highlightIndex === idx
                ? 'bg-cyan-600 text-white ring-2 ring-cyan-400'
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
              }
            `}
            title={`Index: ${idx}, Token: "${token}"`}
          >
            <span className="text-gray-500 text-xs mr-1">{idx}</span>
            {token === ' ' ? '\u2423' : token === '\n' ? '\u21b5' : token}
          </span>
        ))}
        {generatedTokens.map((token, idx) => (
          <span
            key={`gen-${idx}`}
            className="px-2 py-1 rounded text-sm font-mono bg-green-900/50 text-green-300 border border-green-700"
            title={`Generated: "${token}"`}
          >
            {token === ' ' ? '\u2423' : token === '\n' ? '\u21b5' : token}
          </span>
        ))}
      </div>
      <div className="text-xs text-gray-500">
        Click a token to analyze its position
      </div>
    </div>
  );
}
