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
  const data = results.map((r) => {
    // Formatear nombres de casos de manera más compacta
    let name = r.case_name
    // Para benchmarks de modelos: simplificar formato
    name = name.replace(/B=(\d+)_T=(\d+)_K=(\d+)_N=(\d+)/, 'B=$1 T=$2')
    // Para benchmarks de análisis: extraer información clave
    name = name.replace(/Contact Map \(single frame, cutoff=([\d.]+)Å\)/, 'CM single (cutoff=$1Å)')
    name = name.replace(/Contact Map \(batch, (\d+) frames, cutoff=([\d.]+)Å\)/, 'CM batch ($1 frames)')
    name = name.replace(/RMSD \(single frame (\d+) vs 0, with superposition\)/, 'RMSD single (frame $1)')
    name = name.replace(/RMSD \(batch, (\d+) frames vs 0, with superposition\)/, 'RMSD batch ($1 frames)')
    // Truncar si es muy largo
    if (name.length > 30) {
      name = name.substring(0, 27) + '...'
    }
    return {
      name: name,
      baseline: r.baseline_time_ms,
      triton: r.triton_time_ms,
    }
  })

  return (
    <ResponsiveContainer width="100%" height={600}>
      <LineChart 
        data={data}
        margin={{ top: 30, right: 40, bottom: 140, left: 30 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          angle={-60}
          textAnchor="end"
          height={160}
          interval={0}
          tick={{ fontSize: 12, fill: '#6b7280' }}
          dx={-5}
          dy={8}
        />
        <YAxis 
          label={{ value: 'Time (ms)', angle: -90, position: 'left', offset: -5 }} 
          tick={{ fontSize: 13 }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'rgba(255, 255, 255, 0.95)', 
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            padding: '8px 12px'
          }}
        />
        <Legend 
          wrapperStyle={{ paddingTop: '10px' }}
        />
        <Line
          type="monotone"
          dataKey="baseline"
          stroke="#ef4444"
          strokeWidth={2}
          name="Baseline"
          dot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="triton"
          stroke="#0ea5e9"
          strokeWidth={2}
          name="Triton"
          dot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
