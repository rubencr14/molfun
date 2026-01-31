'use client'

import { useState } from 'react'
import { LayoutDashboard, Cpu, FlaskConical, History } from 'lucide-react'

interface SidebarProps {
  kernelType: 'models' | 'analysis'
  onKernelTypeChange: (type: 'models' | 'analysis') => void
  onHistoryClick: () => void
}

export default function Sidebar({ kernelType, onKernelTypeChange, onHistoryClick }: SidebarProps) {
  return (
    <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-screen fixed left-0 top-0 overflow-y-auto">
      <div className="p-6">
        {/* Logo/Title */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            Molfun
          </h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Benchmarks Dashboard
          </p>
        </div>

        {/* Kernel Type Selector */}
        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Kernel Type
          </label>
          <div className="space-y-2">
            <button
              onClick={() => onKernelTypeChange('models')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${
                kernelType === 'models'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <Cpu className="w-5 h-5" />
              Models
            </button>
            <button
              onClick={() => onKernelTypeChange('analysis')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${
                kernelType === 'analysis'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <FlaskConical className="w-5 h-5" />
              Analysis
            </button>
          </div>
        </div>

        {/* History Button */}
        <button
          onClick={onHistoryClick}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
        >
          <History className="w-5 h-5" />
          History
        </button>
      </div>
    </div>
  )
}
