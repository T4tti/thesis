'use client'

import { useLanguage } from '@/context/LanguageContext'
import DataTable from '@/components/DataTable'
import { Database, Building2, Globe2, CalendarRange, AlertCircle } from 'lucide-react'

interface Stats {
  total_records: number
  unique_tickers: number
  unique_sectors: number
  year_range: { min: number; max: number }
}

interface Props {
  stats: Stats | null
  sectors: string[]
  statsError: string | null
}

export default function ReportsClient({ stats, sectors, statsError }: Props) {
  const { t } = useLanguage()
  const tr = t.reports

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* 1. Header: Editorial & Bold */}
      <header className="flex flex-col gap-6 border-b border-gray-200 pb-10 dark:border-gray-800">
        <div className="max-w-3xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-600 dark:border-brand-500/30 dark:bg-brand-500/10 dark:text-brand-300">
            <Database className="h-3.5 w-3.5" />
            {tr.title}
          </div>
          <h1 className="text-4xl font-light tracking-tight text-gray-900 dark:text-white md:text-5xl lg:text-6xl font-serif">
            {tr.title}
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-gray-600 dark:text-gray-300">
            {tr.subtitle}
          </p>
        </div>
      </header>

      {/* Stats error banner */}
      {statsError && (
        <div
          className="flex items-start gap-3 rounded-none border border-red-200 bg-red-50 p-4 dark:border-red-500/30 dark:bg-red-500/10"
          role="alert"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
          <p className="text-sm font-medium text-red-600 dark:text-red-400">{statsError}</p>
        </div>
      )}

      {/* 2. Metrics Section: Unified Analytical Strip */}
      <section className="rounded-none bg-gray-50/50 p-8 shadow-inner ring-1 ring-gray-200/50 dark:bg-gray-800/20 dark:ring-gray-800/50">
        <div className="grid grid-cols-2 gap-y-8 divide-x divide-gray-200 dark:divide-gray-800 md:grid-cols-3">
          <div className="flex flex-col px-6 first:pl-0">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Database className="h-4 w-4" />
              {tr.totalRecords}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              {stats ? stats.total_records.toLocaleString() : '-'}
            </dd>
          </div>
          <div className="flex flex-col px-6">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Building2 className="h-4 w-4" />
              {tr.companies}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-brand-600 dark:text-brand-400">
              {stats ? stats.unique_tickers.toLocaleString() : '-'}
            </dd>
          </div>
          <div className="flex flex-col px-6 last:pr-0 border-r-0">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Globe2 className="h-4 w-4" />
              {tr.sectors}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-ig-dark dark:text-ig">
              {stats ? String(stats.unique_sectors) : '-'}
            </dd>
          </div>
        </div>
      </section>

      {/* 3. Data Table */}
      <section>
        <DataTable sectors={sectors} />
      </section>
    </div>
  )
}
