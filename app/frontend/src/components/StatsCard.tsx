import { ReactNode } from 'react'

interface Props {
  icon: ReactNode
  label: string
  value: string | number
  sub?: string
  accent?: string
}

export default function StatsCard({ icon, label, value, sub, accent = 'text-primary-400' }: Props) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/8 bg-card-gradient bg-surface-200/60 backdrop-blur-sm p-6 group hover:border-primary-500/30 transition-all duration-300 shadow-card">
      {/* Subtle shimmer on hover */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
                      bg-gradient-to-r from-transparent via-white/3 to-transparent -skew-x-12" />

      <div className="relative z-10">
        <div className={`mb-3 w-10 h-10 rounded-xl flex items-center justify-center bg-primary-500/10 ${accent}`}>
          {icon}
        </div>
        <div className={`text-3xl font-bold tracking-tight mb-1 ${accent}`}>{value}</div>
        <div className="text-slate-200 font-medium text-sm">{label}</div>
        {sub && <div className="text-slate-500 text-xs mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}
