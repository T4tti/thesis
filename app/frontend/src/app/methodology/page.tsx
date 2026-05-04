'use client'

import { useEffect, useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import RatingBadge from '@/components/RatingBadge'
import { ChevronDown, ChevronUp, Shield, TrendingUp, AlertTriangle, Cpu, CheckCircle2 } from 'lucide-react'
import { API_BASE_URL } from '@/lib/config'

const API = API_BASE_URL

type RatioKey = keyof typeof import('@/locales/en').en.methodology.ratios

interface BenchmarkData {
  production_model_cv: {
    cv_f1_weighted: number
    cv_f1_macro: number
    cv_balanced_accuracy: number
    n_samples: number
  }
}

const RATIO_KEYS: RatioKey[] = [
  'current_ratio', 'debt_equity_ratio',
  'gross_profit_margin', 'operating_profit_margin', 'ebit_margin', 'pretax_profit_margin', 'net_profit_margin',
  'asset_turnover', 'roe', 'roa',
  'operating_cashflow_ps', 'free_cashflow_ps',
]

export default function MethodologyPage() {
  const { t } = useLanguage()
  const m = t.methodology
  const [openRatio, setOpenRatio] = useState<RatioKey | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null)

  useEffect(() => {
    fetch(`${API}/api/benchmark`).then(r => r.json()).then(setBenchmark).catch(() => {})
  }, [])

  const toggleRatio = (key: RatioKey) =>
    setOpenRatio(prev => (prev === key ? null : key))

  return (
    <div className="space-y-16 animate-in fade-in slide-in-from-bottom-4 duration-700 pb-20">
      <header className="flex flex-col gap-6 border-b border-gray-200 pb-10 dark:border-gray-800">
        <div className="max-w-3xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-600 dark:border-brand-500/30 dark:bg-brand-500/10 dark:text-brand-300">
            <Cpu className="h-3.5 w-3.5" />
            {m.title}
          </div>
          <h1 className="text-4xl font-light tracking-tight text-gray-900 dark:text-white md:text-5xl lg:text-6xl font-serif">
            {m.title}
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-gray-600 dark:text-gray-300">
            {m.subtitle}
          </p>
        </div>
      </header>

      {/* Model & System Features: Asymmetric 12-column grid */}
      <div className="grid grid-cols-1 gap-12 lg:grid-cols-12 lg:gap-16">
        <section className="flex flex-col gap-6 lg:col-span-7">
          <header className="border-b border-gray-900 pb-4 dark:border-white">
            <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white font-serif">{m.systemTitle}</h2>
          </header>
          <div className="space-y-4 pt-2">
            {[
              { rating: 'IG' as const, detail: m.igDetail, bg: 'bg-ig/5', text: 'text-ig-dark dark:text-ig', icon: <Shield className="h-5 w-5" /> },
              { rating: 'HY' as const, detail: m.hyDetail, bg: 'bg-hy/5', text: 'text-hy-dark dark:text-hy', icon: <TrendingUp className="h-5 w-5" /> },
              { rating: 'Distressed' as const, detail: m.distDetail, bg: 'bg-distressed/10', text: 'text-distressed-dark dark:text-distressed', icon: <AlertTriangle className="h-5 w-5" /> },
            ].map(({ rating, detail, bg, text, icon }) => (
              <div key={rating} className={`flex items-start gap-4 rounded-none border-l-2 p-4 ${bg} border-transparent hover:border-current ${text}`}>
                <div className="mt-0.5 shrink-0">{icon}</div>
                <div>
                  <div className="mb-1">
                    <RatingBadge rating={rating} size="sm" />
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-200">{detail}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="flex flex-col gap-6 lg:col-span-5 lg:border-l lg:border-gray-200 lg:pl-12 lg:dark:border-gray-800">
          <header className="border-b border-gray-900 pb-4 dark:border-white">
            <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white font-serif">{m.modelTitle}</h2>
          </header>
          <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-300 pt-2 font-serif italic">
            {m.modelDesc}
          </p>
          <div className="flex flex-wrap gap-3 mt-4">
            {['Median Imputation', 'TLSTMFuzzy', 'Class-Balanced Weights', '5-Fold Stratified CV', 'Point-in-Time Split'].map(tag => (
              <span key={tag} className="flex items-center gap-1.5 rounded-none border border-gray-200 bg-white px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-widest text-gray-500 transition-colors hover:border-brand-500 hover:text-brand-600 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-400">
                <CheckCircle2 className="h-3 w-3" />
                {tag}
              </span>
            ))}
          </div>
        </section>
      </div>

      {/* Ratios Dictionary */}
      <section className="flex flex-col gap-6">
        <header className="border-b border-gray-900 pb-4 dark:border-white">
          <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white font-serif">{m.ratiosTitle}</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{m.ratiosSubtitle}</p>
        </header>

        <div className="mt-2 flex flex-col">
          {RATIO_KEYS.map((key, idx) => {
            const ratio = m.ratios[key]
            const isOpen = openRatio === key
            return (
              <div key={key} className="border-b border-gray-100 dark:border-gray-800/60 last:border-0">
                <button
                  onClick={() => toggleRatio(key)}
                  className="group flex w-full items-center gap-4 py-5 text-left transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/30"
                >
                  <span className="flex w-8 shrink-0 justify-center text-xs font-mono text-brand-500/50 group-hover:text-brand-500 dark:text-brand-300/40 dark:group-hover:text-brand-300">
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className="text-base font-semibold text-gray-900 dark:text-white">{ratio.name}</span>
                    <span className="ml-4 hidden text-xs font-mono text-gray-400 dark:text-gray-500 sm:inline">{ratio.formula}</span>
                  </div>
                  <span className="pr-2">
                    {isOpen
                      ? <ChevronUp className="h-4 w-4 text-brand-500" />
                      : <ChevronDown className="h-4 w-4 text-gray-400 group-hover:text-gray-600 dark:text-gray-500 dark:group-hover:text-gray-400" />}
                  </span>
                </button>
                {isOpen && (
                  <div className="pb-6 pl-12 pr-4 sm:pl-12">
                    <div className="mb-4 inline-flex items-center rounded-none border-b border-gray-200 bg-transparent pb-1 text-xs font-mono text-gray-500 dark:border-gray-700 dark:text-gray-400 sm:hidden">
                      {ratio.formula}
                    </div>
                    <div className="prose prose-sm dark:prose-invert max-w-none font-serif">
                      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">{ratio.interp}</p>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </section>

      {/* Benchmark Strip: Prominent Primary Metric */}
      <section className="flex flex-col gap-6">
        <header className="border-b border-gray-900 pb-4 dark:border-white">
          <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white font-serif">{m.benchmarkTitle}</h2>
        </header>

        <div className="grid grid-cols-1 gap-12 lg:grid-cols-12">
          {/* Primary Metric: F1 Weighted */}
          <div className="flex flex-col items-center justify-center rounded-none border border-brand-200 bg-brand-50/50 p-12 dark:border-brand-500/20 dark:bg-brand-500/5 lg:col-span-7">
            <div className="text-xs font-bold uppercase tracking-widest text-brand-600 dark:text-brand-400 mb-6">
              Primary Model Performance ({m.metricF1w})
            </div>
            <div className="text-7xl font-light tracking-tighter text-brand-600 dark:text-brand-400 font-serif">
              {typeof benchmark?.production_model_cv?.cv_f1_weighted === 'number' 
                ? `${(benchmark.production_model_cv.cv_f1_weighted * 100).toFixed(1)}%` 
                : '...'}
            </div>
            <div className="mt-8 flex items-center gap-4 text-xs font-mono text-gray-500 dark:text-gray-400">
              <span className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3" /> Stratified 5-Fold</span>
              <span className="h-3 w-px bg-gray-300 dark:bg-gray-700"></span>
              <span className="flex items-center gap-1.5">n = {benchmark?.production_model_cv.n_samples?.toLocaleString()}</span>
            </div>
          </div>

          {/* Supporting Metrics */}
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:col-span-5 lg:grid-cols-1">
            <div className="flex flex-col justify-center rounded-none border border-gray-200 bg-white p-8 dark:border-gray-800 dark:bg-gray-900/50">
              <div className="text-xs font-bold uppercase tracking-widest text-gray-500 dark:text-gray-400 mb-2">
                {m.metricF1m}
              </div>
              <div className="text-4xl font-bold tracking-tight text-blue-600 dark:text-blue-400 font-serif">
                {typeof benchmark?.production_model_cv?.cv_f1_macro === 'number' 
                  ? `${(benchmark.production_model_cv.cv_f1_macro * 100).toFixed(1)}%` 
                  : '...'}
              </div>
            </div>
            <div className="flex flex-col justify-center rounded-none border border-gray-200 bg-white p-8 dark:border-gray-800 dark:bg-gray-900/50">
              <div className="text-xs font-bold uppercase tracking-widest text-gray-500 dark:text-gray-400 mb-2">
                {m.metricAcc}
              </div>
              <div className="text-4xl font-bold tracking-tight text-ig-dark dark:text-ig font-serif">
                {typeof benchmark?.production_model_cv?.cv_balanced_accuracy === 'number' 
                  ? `${(benchmark.production_model_cv.cv_balanced_accuracy * 100).toFixed(1)}%` 
                  : '...'}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
