import { useState, useEffect } from 'react';
import StaticDemo from './components/StaticDemo';

interface ScenarioMeta {
  name: string;
  file: string;
  description?: string;
}

export default function StaticApp() {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Try to load scenario index
    const loadIndex = async () => {
      try {
        const response = await fetch('./scenarios/index.json');
        if (response.ok) {
          const data = await response.json();
          setScenarios(data.scenarios || []);
          if (data.scenarios?.length > 0) {
            setSelectedScenario(data.scenarios[0].file);
          }
        }
      } catch {
        // Default scenarios if no index
        setScenarios([
          { name: 'Default', file: './scenarios/default.json' }
        ]);
      } finally {
        setLoading(false);
      }
    };

    loadIndex();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Scenario Selector */}
      {scenarios.length > 1 && (
        <div className="border-b border-gray-700 px-6 py-2">
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">Scenario:</span>
            <select
              value={selectedScenario || ''}
              onChange={(e) => setSelectedScenario(e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm"
            >
              {scenarios.map((s) => (
                <option key={s.file} value={s.file}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Demo */}
      {selectedScenario && (
        <StaticDemo scenarioUrl={selectedScenario} />
      )}
    </div>
  );
}
