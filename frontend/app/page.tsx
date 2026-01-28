'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Play, Loader2, TrendingUp, Clock, CheckCircle2, XCircle, Code, History } from 'lucide-react'
import BenchmarkCard from './components/BenchmarkCard'
import ResultsTable from './components/ResultsTable'
import SpeedupChart from './components/SpeedupChart'
import TimeComparisonChart from './components/TimeComparisonChart'
import CodeViewer from './components/CodeViewer'
import HistoryPanel from './components/HistoryPanel'

const API_BASE_URL = 'http://localhost:8000'

interface Benchmark {
  name: string
  running: boolean
}

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

interface BenchmarkRun {
  benchmark_name: string
  timestamp: string
  results: BenchmarkResult[]
  total_cases: number
  success: boolean
  error?: string
}

interface KernelCode {
  benchmark_name: string
  benchmark_code: string | null
  benchmark_filename: string | null
  kernel_code: string | null
  kernel_filename: string | null
}

export default function Home() {
  const [benchmarks, setBenchmarks] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [runningBenchmark, setRunningBenchmark] = useState<string | null>(null)
  const [currentResults, setCurrentResults] = useState<BenchmarkRun | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isCodeViewerOpen, setIsCodeViewerOpen] = useState(false)
  const [kernelCode, setKernelCode] = useState<KernelCode | null>(null)
  const [loadingCode, setLoadingCode] = useState(false)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)

  useEffect(() => {
    fetchBenchmarks()
  }, [])

  const fetchBenchmarks = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/benchmarks`)
      setBenchmarks(response.data)
      setLoading(false)
    } catch (err) {
      setError('Failed to fetch benchmarks')
      setLoading(false)
    }
  }

  const runBenchmark = async (benchmarkName: string) => {
    setRunningBenchmark(benchmarkName)
    setError(null)
    setCurrentResults(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/benchmarks/${benchmarkName}/run`)
      setCurrentResults(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run benchmark')
    } finally {
      setRunningBenchmark(null)
    }
  }

  const fetchKernelCode = async (benchmarkName: string) => {
    setLoadingCode(true)
    try {
      const response = await axios.get(`${API_BASE_URL}/kernels/${benchmarkName}`)
      setKernelCode(response.data)
      setIsCodeViewerOpen(true)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch kernel code')
    } finally {
      setLoadingCode(false)
    }
  }

  const loadResultFromHistory = async (filename: string) => {
    setError(null)
    setCurrentResults(null)
    try {
      const response = await axios.get(`${API_BASE_URL}/results/${filename}`)
      setCurrentResults(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load result')
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
              Molfun Benchmarks Dashboard
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Professional benchmarking dashboard for Triton kernels
            </p>
          </div>
          <button
            onClick={() => setIsHistoryOpen(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-lg font-medium transition-colors"
          >
            <History className="w-5 h-5" />
            History
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2">
            <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Benchmarks Grid */}
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            {benchmarks.map((benchmark) => (
              <BenchmarkCard
                key={benchmark}
                name={benchmark}
                running={runningBenchmark === benchmark}
                onRun={() => runBenchmark(benchmark)}
              />
            ))}
          </div>
        )}

        {/* Results Section */}
        {currentResults && (
          <div className="mt-8 space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Status</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {currentResults.success ? 'Success' : 'Failed'}
                    </p>
                  </div>
                  {currentResults.success ? (
                    <CheckCircle2 className="w-8 h-8 text-green-500" />
                  ) : (
                    <XCircle className="w-8 h-8 text-red-500" />
                  )}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Total Cases</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {currentResults.total_cases}
                    </p>
                  </div>
                  <Clock className="w-8 h-8 text-primary-500" />
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Avg Speedup</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {currentResults.results.length > 0
                        ? (
                            currentResults.results
                              .filter((r) => r.speedup !== null)
                              .reduce((sum, r) => sum + (r.speedup || 0), 0) /
                            currentResults.results.filter((r) => r.speedup !== null).length
                          ).toFixed(2) + 'x'
                        : 'N/A'}
                    </p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Timestamp</p>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {new Date(currentResults.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* View Code Button */}
            {currentResults.success && (
              <div className="flex justify-end">
                <button
                  onClick={() => fetchKernelCode(currentResults.benchmark_name)}
                  disabled={loadingCode}
                  className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loadingCode ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Code className="w-4 h-4" />
                      View Kernel Code
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Charts */}
            {currentResults.success && currentResults.results.length > 0 && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                      Speedup Comparison
                    </h2>
                    <SpeedupChart results={currentResults.results} />
                  </div>

                  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                      Time Comparison
                    </h2>
                    <TimeComparisonChart results={currentResults.results} />
                  </div>
                </div>

                {/* Results Table */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                  <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                    Detailed Results
                  </h2>
                  <ResultsTable results={currentResults.results} />
                </div>
              </>
            )}

            {/* Error Display */}
            {!currentResults.success && currentResults.error && (
              <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border border-red-200 dark:border-red-800">
                <h3 className="text-lg font-semibold text-red-900 dark:text-red-200 mb-2">
                  Error Details
                </h3>
                <pre className="text-sm text-red-800 dark:text-red-300 whitespace-pre-wrap">
                  {currentResults.error}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Code Viewer Panel */}
      {kernelCode && (
        <CodeViewer
          isOpen={isCodeViewerOpen}
          onClose={() => setIsCodeViewerOpen(false)}
          benchmarkCode={kernelCode.benchmark_code}
          benchmarkFilename={kernelCode.benchmark_filename}
          kernelCode={kernelCode.kernel_code}
          kernelFilename={kernelCode.kernel_filename}
        />
      )}

      {/* History Panel */}
      <HistoryPanel
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectResult={loadResultFromHistory}
      />
    </main>
  )
}
