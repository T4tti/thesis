'use client'

import { useEffect, useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import RatingBadge from '@/components/RatingBadge'
import { ChevronDown, ChevronUp, Shield, TrendingUp, AlertTriangle, Cpu, CheckCircle2 } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

      {/* Header */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary-500/30 bg-primary-500/10 text-primary-300 text-sm font-medium mb-6">
          <Cpu className="w-3.5 h-3.5" />
          {m.title}
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-4">{m.title}</h1>
        <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">{m.subtitle}</p>
      </div>

      {/* ── 3-Group system ──────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold text-white mb-2">{m.systemTitle}</h2>
        <p className="text-slate-400 mb-8 leading-relaxed">{m.systemDesc}</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[
            { rating: 'IG' as const,         detail: m.igDetail,   border:'border-ig/30',           bg:'bg-ig-gradient',   icon:<Shield className="w-5 h-5 text-ig" />,           hover:'hover:border-ig/60' },
            { rating: 'HY' as const,         detail: m.hyDetail,   border:'border-hy/30',           bg:'bg-hy-gradient',   icon:<TrendingUp className="w-5 h-5 text-hy" />,       hover:'hover:border-hy/60' },
            { rating: 'Distressed' as const, detail: m.distDetail, border:'border-distressed/30',   bg:'bg-dist-gradient', icon:<AlertTriangle className="w-5 h-5 text-distressed" />, hover:'hover:border-distressed/60' },
          ].map(({ rating, detail, border, bg, icon, hover }) => (
            <div key={rating} className={`rounded-2xl border ${border} ${bg} ${hover} p-6 transition-all duration-300`}>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-lg bg-surface-300/50 flex items-center justify-center">
                  {icon}
                </div>
                <RatingBadge rating={rating} size="md" />
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">{detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── 12 Financial Ratios ─────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold text-white mb-2">{m.ratiosTitle}</h2>
        <p className="text-slate-400 mb-8 leading-relaxed">{m.ratiosSubtitle}</p>

        <div className="space-y-2">
          {RATIO_KEYS.map((key, idx) => {
            const ratio = m.ratios[key]
            const isOpen = openRatio === key
            return (
              <div
                key={key}
                className={`rounded-xl border transition-all duration-200 overflow-hidden
                  ${isOpen ? 'border-primary-500/40 bg-primary-500/5' : 'border-white/8 bg-surface-200/40 hover:border-white/15'}`}
              >
                <button
                  onClick={() => toggleRatio(key)}
                  className="w-full flex items-center gap-4 px-5 py-4 text-left group"
                >
                  <span className="w-7 h-7 rounded-lg bg-surface-100/40 flex items-center justify-center text-xs font-mono text-primary-400 shrink-0">
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className="font-semibold text-white">{ratio.name}</span>
                    <span className="ml-3 text-slate-500 text-xs font-mono hidden sm:inline">{ratio.formula}</span>
                  </div>
                  {isOpen
                    ? <ChevronUp className="w-4 h-4 text-primary-400 shrink-0" />
                    : <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />}
                </button>
                {isOpen && (
                  <div className="px-5 pb-5 pl-16">
                    <div className="flex items-center gap-2 mb-2 text-xs font-mono text-slate-500 bg-surface-300/50 px-3 py-1.5 rounded-lg w-fit">
                      {ratio.formula}
                    </div>
                    <p className="text-slate-300 text-sm leading-relaxed">{ratio.interp}</p>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Model pipeline ──────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold text-white mb-2">{m.modelTitle}</h2>
        <p className="text-slate-400 mb-8 leading-relaxed">{m.modelDesc}</p>

        <div className="flex flex-wrap gap-2 mb-4">
          {['Median Imputation', 'LightGBM (GBDT)', 'Class-Balanced Weights', '5-Fold Stratified CV', 'Point-in-Time Split'].map(tag => (
            <span key={tag} className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/25 text-primary-300 text-xs font-medium">
              <CheckCircle2 className="w-3 h-3" />
              {tag}
            </span>
          ))}
        </div>
      </section>

      {/* ── Benchmark ───────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">{m.benchmarkTitle}</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { label: m.metricF1w,  key: 'cv_f1_weighted',      accent: 'text-primary-400' },
            { label: m.metricF1m,  key: 'cv_f1_macro',         accent: 'text-sky-400'     },
            { label: m.metricAcc,  key: 'cv_balanced_accuracy', accent: 'text-ig-light'   },
          ].map(({ label, key, accent }) => {
            const val = benchmark?.production_model_cv?.[key as keyof typeof benchmark.production_model_cv]
            return (
              <div key={key} className="glass-card rounded-2xl p-6 text-center">
                <div className={`text-4xl font-bold mb-2 ${accent}`}>
                  {typeof val === 'number' ? `${(val * 100).toFixed(1)}%` : '…'}
                </div>
                <div className="text-slate-400 text-sm">{label}</div>
              </div>
            )
          })}
        </div>
        {benchmark && (
          <p className="text-slate-500 text-xs text-center mt-4">
            n = {benchmark.production_model_cv.n_samples?.toLocaleString()} samples · 5-Fold Stratified CV
          </p>
        )}
      </section>
    </div>
  )
}
