'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { X, Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'

interface HistoryItem {
  filename: string
  benchmark_name: string
  timestamp: string
  total_cases: number
  success: boolean
}

interface HistoryPanelProps {
  isOpen: boolean
  onClose: () => void
  onSelectResult: (filename: string) => void
}

export default function HistoryPanel({ isOpen, onClose, onSelectResult }: HistoryPanelProps) {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      fetchHistory()
    }
  }, [isOpen])

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get(`${API_BASE_URL}/results`)
      setHistory(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch history')
    } finally {
      setLoading(false)
    }
  }

  const handleSelect = async (filename: string) => {
    onSelectResult(filename)
    onClose()
  }

  if (!isOpen) return null

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed right-0 top-0 h-full w-full md:w-96 bg-white dark:bg-gray-900 z-50 shadow-2xl transform transition-transform duration-300 ease-in-out">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Run History
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="h-[calc(100%-80px)] overflow-auto p-4">
          {loading ? (
            <div className="flex justify-center items-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
            </div>
          ) : error ? (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
            </div>
          ) : history.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">No history available</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <button
                  key={item.filename}
                  onClick={() => handleSelect(item.filename)}
                  className="w-full text-left p-4 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 dark:text-white capitalize mb-1">
                        {item.benchmark_name.replace(/_/g, ' ')}
                      </h3>
                      <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <Clock className="w-4 h-4" />
                        <span>{new Date(item.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                    {item.success ? (
                      <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 ml-2" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 ml-2" />
                    )}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {item.total_cases} case{item.total_cases !== 1 ? 's' : ''}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
