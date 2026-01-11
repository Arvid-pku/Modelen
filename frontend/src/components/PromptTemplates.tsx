import { useState, useMemo } from 'react';
import { useTheme } from '../contexts/ThemeContext';

interface PromptTemplate {
  id: string;
  name: string;
  category: string;
  prompt: string;
  description: string;
  suggestedInterventions?: string;
}

const TEMPLATES: PromptTemplate[] = [
  // Factual Recall
  {
    id: 'capital-france',
    name: 'Capital of France',
    category: 'Factual Recall',
    prompt: 'The capital of France is',
    description: 'Classic factual recall - watch how "Paris" emerges through layers',
  },
  {
    id: 'capital-japan',
    name: 'Capital of Japan',
    category: 'Factual Recall',
    prompt: 'The capital of Japan is',
    description: 'Factual recall with different entity',
  },
  {
    id: 'inventor-telephone',
    name: 'Inventor of Telephone',
    category: 'Factual Recall',
    prompt: 'The telephone was invented by Alexander Graham',
    description: 'Name completion with factual knowledge',
  },

  // Induction Heads
  {
    id: 'induction-simple',
    name: 'Simple Repetition',
    category: 'Induction Heads',
    prompt: 'The cat sat on the mat. The cat sat on the',
    description: 'Test induction heads - model should predict "mat"',
  },
  {
    id: 'induction-abc',
    name: 'ABC Pattern',
    category: 'Induction Heads',
    prompt: 'A B C D E F G A B C D E F',
    description: 'Pattern completion - should predict "G"',
  },
  {
    id: 'induction-names',
    name: 'Name Repetition',
    category: 'Induction Heads',
    prompt: 'John went to the store. Mary stayed home. John went to the',
    description: 'Complex induction with multiple entities',
  },

  // Syntax & Grammar
  {
    id: 'subject-verb',
    name: 'Subject-Verb Agreement',
    category: 'Syntax',
    prompt: 'The keys to the cabinet',
    description: 'Test number agreement - should predict "are" not "is"',
  },
  {
    id: 'reflexive',
    name: 'Reflexive Pronoun',
    category: 'Syntax',
    prompt: 'The queen told the princess that she admired',
    description: 'Ambiguous pronoun reference resolution',
  },
  {
    id: 'garden-path',
    name: 'Garden Path Sentence',
    category: 'Syntax',
    prompt: 'The horse raced past the barn',
    description: 'Garden path - "fell" is surprising but grammatical',
  },

  // Reasoning
  {
    id: 'greater-than',
    name: 'Number Comparison',
    category: 'Reasoning',
    prompt: 'Which is greater, 9.11 or 9.9? The answer is',
    description: 'Tests numerical reasoning (models often fail this)',
  },
  {
    id: 'negation',
    name: 'Negation Understanding',
    category: 'Reasoning',
    prompt: 'A person who is not alive is',
    description: 'Tests negation processing',
  },

  // Indirect Object Identification (IOI)
  {
    id: 'ioi-simple',
    name: 'IOI - Simple',
    category: 'IOI Task',
    prompt: 'When Mary and John went to the store, John gave a drink to',
    description: 'Classic IOI - should predict "Mary"',
  },
  {
    id: 'ioi-reversed',
    name: 'IOI - Reversed',
    category: 'IOI Task',
    prompt: 'When John and Mary went to the store, Mary gave a drink to',
    description: 'IOI with reversed names - should predict "John"',
  },

  // Multilingual
  {
    id: 'french',
    name: 'French Completion',
    category: 'Multilingual',
    prompt: 'Bonjour, je m\'appelle',
    description: 'French name completion',
  },
  {
    id: 'code-python',
    name: 'Python Code',
    category: 'Code',
    prompt: 'def fibonacci(n):\n    if n <= 1:\n        return',
    description: 'Code completion - should predict "n"',
  },
];

interface PromptTemplatesProps {
  onSelect: (prompt: string) => void;
}

export default function PromptTemplates({ onSelect }: PromptTemplatesProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [isOpen, setIsOpen] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const colors = useMemo(() => ({
    bgSecondary: isDark ? '#1f2937' : '#ffffff',
    bgTertiary: isDark ? '#374151' : '#e5e7eb',
    textPrimary: isDark ? '#f3f4f6' : '#111827',
    textSecondary: isDark ? '#9ca3af' : '#4b5563',
    borderColor: isDark ? '#374151' : '#d1d5db',
  }), [isDark]);

  const categories = [...new Set(TEMPLATES.map(t => t.category))];
  const filteredTemplates = selectedCategory
    ? TEMPLATES.filter(t => t.category === selectedCategory)
    : TEMPLATES;

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded text-sm font-medium transition-colors"
        style={{ backgroundColor: colors.bgTertiary, color: colors.textPrimary }}
      >
        <span>Prompt Templates</span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div
          className="mt-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden"
          style={{ backgroundColor: colors.bgSecondary, borderColor: colors.borderColor }}
        >
          {/* Category filter */}
          <div className="p-2 border-b border-gray-200 dark:border-gray-700 flex flex-wrap gap-1" style={{ borderColor: colors.borderColor }}>
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-2 py-1 text-xs rounded ${
                selectedCategory === null
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
              style={selectedCategory !== null ? { backgroundColor: colors.bgTertiary, color: colors.textPrimary } : undefined}
            >
              All
            </button>
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2 py-1 text-xs rounded ${
                  selectedCategory === cat
                    ? 'bg-cyan-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
                style={selectedCategory !== cat ? { backgroundColor: colors.bgTertiary, color: colors.textPrimary } : undefined}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Template list */}
          <div className="max-h-64 overflow-y-auto">
            {filteredTemplates.map(template => (
              <button
                key={template.id}
                onClick={() => {
                  onSelect(template.prompt);
                  setIsOpen(false);
                }}
                className="w-full text-left p-3 hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700 last:border-0 transition-colors"
                style={{ borderColor: colors.borderColor }}
              >
                <div className="font-medium text-sm" style={{ color: colors.textPrimary }}>{template.name}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1" style={{ color: colors.textSecondary }}>
                  {template.description}
                </div>
                <div className="text-xs font-mono text-cyan-600 dark:text-cyan-400 mt-1 truncate">
                  "{template.prompt}"
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
