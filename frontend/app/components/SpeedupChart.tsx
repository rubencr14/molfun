'use client'

import {
  ComposedChart,
  Bar,
  ReferenceLine,
  ReferenceArea,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'

interface BenchmarkResult {
  case_name: string
  speedup: number | null
}

interface SpeedupChartProps {
  results: BenchmarkResult[]
}

export default function SpeedupChart({ results }: SpeedupChartProps) {
  const data = results
    .filter((r) => r.speedup !== null)
    .map((r) => {
      const speedup = r.speedup || 1
      return {
        name: r.case_name.replace(/B=(\d+)_T=(\d+)_K=(\d+)_N=(\d+)/, 'B=$1 T=$2'),
        speedup: speedup,
      }
    })

  // Calcular el máximo y mínimo para el dominio del eje Y
  const maxSpeedup = Math.max(...data.map((d) => d.speedup), 1.5)
  const minSpeedup = Math.min(...data.map((d) => d.speedup), 0.5)
  // Asegurar que el mínimo no sea menor que 0.1 para evitar números extraños
  const yMin = Math.max(0.1, minSpeedup - 0.2)
  const yMax = maxSpeedup + 0.2

  // Función para determinar el color de la barra
  const getBarColor = (speedup: number) => {
    return speedup >= 1 ? '#22c55e' : '#ef4444'
  }

  // Formatear los ticks del eje Y para ocultar valores muy pequeños o 0
  const formatYAxisTick = (value: number) => {
    if (value < 0.2) return ''
    return value.toFixed(1)
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data}>
        <defs>
          <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" stopOpacity={0.15} />
            <stop offset="100%" stopColor="#22c55e" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.05} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.15} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          angle={-45}
          textAnchor="end"
          height={100}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          label={{ value: 'Speedup (x)', angle: -90, position: 'left', offset: -5 }}
          domain={[yMin, yMax]}
          tickFormatter={formatYAxisTick}
        />
        <Tooltip />
        <Legend
          content={({ payload }) => (
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm text-gray-700 dark:text-gray-300">Improvement (≥1x)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm text-gray-700 dark:text-gray-300">Regression (&lt;1x)</span>
              </div>
            </div>
          )}
        />
        {/* Área verde por encima de 1 */}
        <ReferenceArea y1={1} y2={yMax} fill="url(#greenGradient)" stroke="none" />
        {/* Área roja por debajo de 1 */}
        <ReferenceArea y1={yMin} y2={1} fill="url(#redGradient)" stroke="none" />
        {/* Línea de referencia en y=1 */}
        <ReferenceLine
          y={1}
          stroke="#6b7280"
          strokeWidth={2}
          strokeDasharray="5 5"
          label={{ value: 'Baseline (1x)', position: 'topRight', fill: '#6b7280' }}
        />
        {/* Barras de speedup con colores dinámicos */}
        <Bar dataKey="speedup" name="Speedup" radius={[4, 4, 0, 0]}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getBarColor(entry.speedup)} />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  )
}
