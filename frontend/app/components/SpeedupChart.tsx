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
    <ResponsiveContainer width="100%" height={600}>
      <ComposedChart 
        data={data}
        margin={{ top: 30, right: 40, bottom: 140, left: 30 }}
      >
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
          angle={-60}
          textAnchor="end"
          height={160}
          interval={0}
          tick={{ fontSize: 12, fill: '#6b7280' }}
          dx={-5}
          dy={8}
        />
        <YAxis
          label={{ value: 'Speedup (x)', angle: -90, position: 'left', offset: -5 }}
          domain={[yMin, yMax]}
          tickFormatter={formatYAxisTick}
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
          label={{ 
            value: 'Baseline (1x)', 
            position: 'insideTopRight', 
            fill: '#6b7280',
            fontSize: 11,
            offset: 10
          }}
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
