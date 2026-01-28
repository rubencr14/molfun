import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Molfun Benchmarks Dashboard',
  description: 'Professional benchmarking dashboard for Triton kernels',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
