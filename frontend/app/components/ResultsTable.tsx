'use client'

interface BenchmarkResult {
  benchmark_name: string
  case_name: string
  baseline_time_ms: number
  triton_time_ms: number
  speedup: number | null
  max_diff: number
  mean_diff: number
  metadata: Record<string, any>
}

interface ResultsTableProps {
  results: BenchmarkResult[]
}

export default function ResultsTable({ results }: ResultsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">Case</th>
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Baseline (ms)
            </th>
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Triton (ms)
            </th>
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Speedup
            </th>
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Max Diff
            </th>
            <th className="p-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Mean Diff
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, idx) => (
            <tr
              key={idx}
              className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50"
            >
              <td className="p-3 text-sm text-gray-900 dark:text-gray-100">
                {result.case_name}
              </td>
              <td className="p-3 text-sm text-gray-700 dark:text-gray-300">
                {result.baseline_time_ms.toFixed(4)}
              </td>
              <td className="p-3 text-sm text-gray-700 dark:text-gray-300">
                {result.triton_time_ms.toFixed(4)}
              </td>
              <td className="p-3 text-sm">
                {result.speedup !== null ? (
                  <span
                    className={`font-semibold ${
                      result.speedup >= 1.5
                        ? 'text-green-600 dark:text-green-400'
                        : result.speedup >= 1.1
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : result.speedup >= 1
                        ? 'text-gray-600 dark:text-gray-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {result.speedup.toFixed(2)}x
                  </span>
                ) : (
                  <span className="text-gray-400">N/A</span>
                )}
              </td>
              <td className="p-3 text-sm text-gray-700 dark:text-gray-300">
                {result.max_diff.toExponential(2)}
              </td>
              <td className="p-3 text-sm text-gray-700 dark:text-gray-300">
                {result.mean_diff.toExponential(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
