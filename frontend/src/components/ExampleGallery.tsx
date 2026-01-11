import { useState } from 'react';

interface Example {
  id: string;
  title: string;
  category: string;
  description: string;
  prompt: string;
  model: string;
  phenomenon: string;
  highlights: string[];
  imageUrl?: string;
}

const EXAMPLES: Example[] = [
  {
    id: 'induction-heads',
    title: 'Induction Heads in Action',
    category: 'Attention Patterns',
    description: 'Watch how the model copies patterns from earlier in the sequence. In layer 5-6, you\'ll see attention heads that look at "cat sat" when predicting after "The cat sat".',
    prompt: 'The cat sat on the mat. The cat sat on the',
    model: 'GPT-2',
    phenomenon: 'Induction Heads',
    highlights: [
      'Look at attention in layers 5-6',
      'Head attends to previous occurrence',
      'Predicts "mat" with high confidence',
    ],
  },
  {
    id: 'factual-recall',
    title: 'Factual Knowledge Emergence',
    category: 'Logit Lens',
    description: 'See how factual knowledge emerges through layers. Early layers show general tokens, middle layers narrow down to locations, and final layers confidently predict "Paris".',
    prompt: 'The capital of France is',
    model: 'GPT-2',
    phenomenon: 'Knowledge Retrieval',
    highlights: [
      'Early layers: generic predictions',
      'Middle layers: geography-related tokens appear',
      'Final layers: "Paris" dominates',
    ],
  },
  {
    id: 'subject-verb',
    title: 'Subject-Verb Agreement',
    category: 'Syntax Processing',
    description: 'The model tracks the subject "keys" (plural) across an intervening phrase to predict "are" instead of "is". Watch attention flow back to the subject.',
    prompt: 'The keys to the cabinet',
    model: 'GPT-2',
    phenomenon: 'Long-range Dependencies',
    highlights: [
      'Subject is "keys" (plural)',
      'Distractor "cabinet" is singular',
      'Model correctly predicts "are"',
    ],
  },
  {
    id: 'ioi-task',
    title: 'Indirect Object Identification',
    category: 'Circuit Analysis',
    description: 'A classic interpretability benchmark. The model identifies that John is the giver and Mary is the receiver, so "Mary" is predicted as the indirect object.',
    prompt: 'When Mary and John went to the store, John gave a drink to',
    model: 'GPT-2',
    phenomenon: 'IOI Circuit',
    highlights: [
      'Duplicate token heads identify "Mary" and "John"',
      'S-inhibition heads suppress the subject (John)',
      'Name mover heads promote "Mary"',
    ],
  },
  {
    id: 'layer-skip',
    title: 'Layer Importance via Skipping',
    category: 'Interventions',
    description: 'Skip individual layers to see which ones are most important for the prediction. Middle layers often contribute most to factual recall.',
    prompt: 'The Eiffel Tower is located in',
    model: 'GPT-2',
    phenomenon: 'Ablation Studies',
    highlights: [
      'Skip layer 5: prediction changes',
      'Skip early layers: minimal impact',
      'Skip final layer: output degraded',
    ],
  },
  {
    id: 'numerical',
    title: 'Number Processing Failure',
    category: 'Reasoning Limits',
    description: 'Models often struggle with decimal comparison. See how the model incorrectly processes 9.11 vs 9.9, revealing limitations in numerical reasoning.',
    prompt: 'Which is greater, 9.11 or 9.9? The answer is',
    model: 'GPT-2',
    phenomenon: 'Reasoning Failure',
    highlights: [
      'Model often predicts "9.11"',
      'Attention focuses on digit count',
      'Missing decimal understanding',
    ],
  },
];

interface ExampleGalleryProps {
  onSelectExample: (prompt: string) => void;
  isOpen: boolean;
  onClose: () => void;
}

export default function ExampleGallery({ onSelectExample, isOpen, onClose }: ExampleGalleryProps) {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedExample, setExpandedExample] = useState<string | null>(null);

  if (!isOpen) return null;

  const categories = [...new Set(EXAMPLES.map(e => e.category))];
  const filteredExamples = selectedCategory
    ? EXAMPLES.filter(e => e.category === selectedCategory)
    : EXAMPLES;

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />

      {/* Gallery Modal */}
      <div className="fixed inset-4 md:inset-10 z-50 bg-white dark:bg-gray-900 rounded-lg shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-cyan-600 to-blue-600 text-white px-6 py-4 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">Example Gallery</h2>
              <p className="text-sm text-white/80 mt-1">
                Pre-computed examples showing interesting model behaviors
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Category Filter */}
        <div className="px-6 py-3 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                selectedCategory === null
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
              }`}
            >
              All
            </button>
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  selectedCategory === cat
                    ? 'bg-cyan-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Example Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredExamples.map(example => (
              <div
                key={example.id}
                className={`bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 transition-all ${
                  expandedExample === example.id ? 'ring-2 ring-cyan-500' : 'hover:border-cyan-500'
                }`}
              >
                {/* Card Header */}
                <div className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <span className="px-2 py-0.5 bg-cyan-100 dark:bg-cyan-900 text-cyan-700 dark:text-cyan-300 text-xs rounded">
                      {example.phenomenon}
                    </span>
                    <span className="text-xs text-gray-500">{example.model}</span>
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    {example.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                    {example.description}
                  </p>
                </div>

                {/* Prompt Preview */}
                <div className="px-4 pb-3">
                  <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono text-gray-700 dark:text-gray-300 truncate">
                    "{example.prompt}"
                  </div>
                </div>

                {/* Expand/Collapse */}
                {expandedExample === example.id && (
                  <div className="px-4 pb-4 border-t border-gray-200 dark:border-gray-700">
                    <h4 className="text-sm font-medium mt-3 mb-2">What to look for:</h4>
                    <ul className="space-y-1">
                      {example.highlights.map((highlight, idx) => (
                        <li key={idx} className="text-sm text-gray-600 dark:text-gray-400 flex items-start gap-2">
                          <span className="text-cyan-500 mt-1">•</span>
                          {highlight}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Actions */}
                <div className="px-4 py-3 bg-gray-100 dark:bg-gray-700/50 flex justify-between items-center">
                  <button
                    onClick={() => setExpandedExample(expandedExample === example.id ? null : example.id)}
                    className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    {expandedExample === example.id ? 'Show less' : 'Learn more'}
                  </button>
                  <button
                    onClick={() => {
                      onSelectExample(example.prompt);
                      onClose();
                    }}
                    className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-700 text-white text-sm rounded transition-colors"
                  >
                    Try this prompt
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
