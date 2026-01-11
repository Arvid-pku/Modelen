import { useMemoryInfo, useClearMemory } from '../hooks/useApi';

export default function MemoryDisplay() {
  const { data: memoryInfo, isLoading } = useMemoryInfo();
  const clearMemory = useClearMemory();

  if (isLoading || !memoryInfo) {
    return null;
  }

  if (!memoryInfo.gpu_available) {
    return (
      <div className="text-xs text-gray-500 dark:text-gray-400">
        CPU Mode
      </div>
    );
  }

  const device = memoryInfo.devices[0];
  if (!device) return null;

  const utilizationColor = device.utilization_percent > 80
    ? 'bg-red-500'
    : device.utilization_percent > 50
    ? 'bg-yellow-500'
    : 'bg-green-500';

  return (
    <div className="flex items-center gap-2">
      <div className="flex flex-col items-end text-xs">
        <span className="text-gray-600 dark:text-gray-400">
          {device.name.split(' ').slice(0, 2).join(' ')}
        </span>
        <span className="text-gray-500 dark:text-gray-500 font-mono">
          {device.allocated_memory_gb.toFixed(1)}/{device.total_memory_gb.toFixed(1)} GB
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-16 h-6 bg-gray-300 dark:bg-gray-700 rounded overflow-hidden flex flex-col justify-center">
        <div
          className={`h-2 ${utilizationColor} transition-all duration-300`}
          style={{ width: `${Math.min(device.utilization_percent, 100)}%` }}
        />
        <span className="text-[10px] text-center text-gray-600 dark:text-gray-400">
          {device.utilization_percent.toFixed(0)}%
        </span>
      </div>

      {/* Clear button */}
      <button
        onClick={() => clearMemory.mutate()}
        disabled={clearMemory.isPending}
        className="p-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded text-gray-600 dark:text-gray-300 transition-colors"
        title="Clear GPU cache"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>
  );
}
