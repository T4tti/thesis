'use client'

import { useEffect, useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import DataTable from '@/components/DataTable'
import { Database, Building2, Globe2, CalendarRange } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Stats {
  total_records: number
  unique_tickers: number
  unique_sectors: number
  year_range: { min: number; max: number }
}

export default function ReportsPage() {
  const { t } = useLanguage()
  const tr = t.reports
  const [stats, setStats] = useState<Stats | null>(null)
  const [sectors, setSectors] = useState<string[]>([])

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/stats`).then(r => r.json()),
      fetch(`${API}/api/sectors`).then(r => r.json()),
    ]).then(([s, sec]) => {
      setStats(s)
      setSectors(sec)
    }).catch(() => {})
  }, [])

  const statItems = [
    {
      label: tr.totalRecords,
      value: stats ? stats.total_records.toLocaleString() : '—',
      icon: <Database className="w-5 h-5" />,
      accent: 'text-primary-400',
    },
    {
      label: tr.companies,
      value: stats ? stats.unique_tickers.toLocaleString() : '—',
      icon: <Building2 className="w-5 h-5" />,
      accent: 'text-sky-400',
    },
    {
      label: tr.sectors,
      value: stats ? String(stats.unique_sectors) : '—',
      icon: <Globe2 className="w-5 h-5" />,
      accent: 'text-ig-light',
    },
    {
      label: tr.dateRange,
      value: stats ? `${stats.year_range.min}–${stats.year_range.max}` : '—',
      icon: <CalendarRange className="w-5 h-5" />,
      accent: 'text-hy-light',
    },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-2 text-primary-400 text-sm font-medium mb-3">
          <Database className="w-4 h-4" />
          <span className="uppercase tracking-widest text-xs">{tr.title}</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-3">{tr.title}</h1>
        <p className="text-slate-400 max-w-xl">{tr.subtitle}</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        {statItems.map(({ label, value, icon, accent }) => (
          <div key={label} className="glass-card rounded-xl px-5 py-4 flex items-center gap-4">
            <div className={`${accent} shrink-0`}>{icon}</div>
            <div>
              <div className={`text-2xl font-bold ${accent}`}>{value}</div>
              <div className="text-slate-400 text-xs mt-0.5">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Data table */}
      <DataTable sectors={sectors} />
    </div>
  )
}
