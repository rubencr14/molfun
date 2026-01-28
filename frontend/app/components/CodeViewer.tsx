'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { X } from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodeViewerProps {
  isOpen: boolean
  onClose: () => void
  benchmarkCode: string | null
  benchmarkFilename: string | null
  kernelCode: string | null
  kernelFilename: string | null
}

export default function CodeViewer({
  isOpen,
  onClose,
  benchmarkCode,
  benchmarkFilename,
  kernelCode,
  kernelFilename,
}: CodeViewerProps) {
  const [width, setWidth] = useState(50) // Porcentaje del viewport (50% por defecto)
  const [isResizing, setIsResizing] = useState(false)
  const [activeTab, setActiveTab] = useState<'benchmark' | 'kernel'>('benchmark')
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
      // Establecer el tab activo por defecto
      if (benchmarkCode) {
        setActiveTab('benchmark')
      } else if (kernelCode) {
        setActiveTab('kernel')
      }
    } else {
      document.body.style.overflow = 'unset'
    }
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen, benchmarkCode, kernelCode])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return

      const newWidth = ((window.innerWidth - e.clientX) / window.innerWidth) * 100
      // Limitar entre 30% y 90% del viewport
      const clampedWidth = Math.min(Math.max(newWidth, 30), 90)
      setWidth(clampedWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing])

  if (!isOpen) return null

  const currentCode = activeTab === 'benchmark' ? benchmarkCode : kernelCode
  const currentFilename = activeTab === 'benchmark' ? benchmarkFilename : kernelFilename

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Side Panel */}
      <div
        ref={panelRef}
        className="fixed right-0 top-0 h-full bg-white dark:bg-gray-900 z-50 shadow-2xl transform transition-transform duration-300 ease-in-out"
        style={{ width: `${width}%` }}
      >
        {/* Resize Handle */}
        <div
          onMouseDown={handleMouseDown}
          className={`absolute left-0 top-0 h-full w-1 cursor-col-resize bg-gray-300 dark:bg-gray-600 hover:bg-primary-500 transition-colors ${
            isResizing ? 'bg-primary-500' : ''
          }`}
          style={{ zIndex: 10 }}
        />

        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Code Viewer
            </h2>
            {/* Tabs */}
            {(benchmarkCode || kernelCode) && (
              <div className="flex gap-2">
                {benchmarkCode && (
                  <button
                    onClick={() => setActiveTab('benchmark')}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                      activeTab === 'benchmark'
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                    }`}
                  >
                    Benchmark
                  </button>
                )}
                {kernelCode && (
                  <button
                    onClick={() => setActiveTab('kernel')}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                      activeTab === 'kernel'
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                    }`}
                  >
                    Kernel
                  </button>
                )}
              </div>
            )}
            {currentFilename && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {currentFilename}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors ml-4"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>
        </div>

        {/* Code Content */}
        <div className="h-[calc(100%-120px)] overflow-auto bg-gray-900">
          {currentCode ? (
            <SyntaxHighlighter
              language="python"
              style={atomDark}
              customStyle={{
                margin: 0,
                padding: '1rem',
                height: '100%',
                fontSize: '0.875rem',
                lineHeight: '1.5',
                background: 'transparent',
              }}
              showLineNumbers
              lineNumberStyle={{
                minWidth: '3em',
                paddingRight: '1em',
                color: '#6b7280',
              }}
            >
              {currentCode}
            </SyntaxHighlighter>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              No code available
            </div>
          )}
        </div>
      </div>
    </>
  )
}
