'use client'

import { Play, Loader2 } from 'lucide-react'

interface BenchmarkCardProps {
  name: string
  running: boolean
  onRun: () => void
}

export default function BenchmarkCard({ name, running, onRun }: BenchmarkCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
            {name.replace(/_/g, ' ')}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Click to run benchmark
          </p>
        </div>
        <button
          onClick={onRun}
          disabled={running}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            running
              ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
              : 'bg-primary-600 hover:bg-primary-700 text-white'
          }`}
        >
          {running ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run
            </>
          )}
        </button>
      </div>
    </div>
  )
}
