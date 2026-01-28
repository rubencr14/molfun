'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface BenchmarkResult {
  case_name: string
  baseline_time_ms: number
  triton_time_ms: number
}

interface TimeComparisonChartProps {
  results: BenchmarkResult[]
}

export default function TimeComparisonChart({ results }: TimeComparisonChartProps) {
  const data = results.map((r) => ({
    name: r.case_name.replace(/B=(\d+)_T=(\d+)_K=(\d+)_N=(\d+)/, 'B=$1 T=$2'),
    baseline: r.baseline_time_ms,
    triton: r.triton_time_ms,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          angle={-45}
          textAnchor="end"
          height={100}
          tick={{ fontSize: 12 }}
        />
        <YAxis label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="baseline"
          stroke="#ef4444"
          strokeWidth={2}
          name="Baseline"
        />
        <Line
          type="monotone"
          dataKey="triton"
          stroke="#0ea5e9"
          strokeWidth={2}
          name="Triton"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
