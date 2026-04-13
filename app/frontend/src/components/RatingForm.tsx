'use client'

import { useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import { Send, RotateCcw, AlertCircle } from 'lucide-react'
import ProbabilityBars from './ProbabilityBars'
import RatingBadge from './RatingBadge'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const FIELD_GROUPS = [
  {
    key: 'groupLiquidity',
    fields: ['current_ratio', 'debt_equity_ratio'],
  },
  {
    key: 'groupProfitability',
    fields: ['gross_profit_margin', 'operating_profit_margin', 'ebit_margin', 'pretax_profit_margin', 'net_profit_margin'],
  },
  {
    key: 'groupReturns',
    fields: ['asset_turnover', 'roe', 'roa'],
  },
  {
    key: 'groupCashflow',
    fields: ['operating_cashflow_ps', 'free_cashflow_ps'],
  },
] as const

interface PredictResult {
  rating: string
  probabilities: Record<string, number>
  confidence: number
  risk_level: string
  risk_score: number
  interpretation_en: string
  interpretation_vi: string
  label_en: string
  label_vi: string
  color: string
}

type FieldKey = typeof FIELD_GROUPS[number]['fields'][number]

export default function RatingForm() {
  const { t, lang } = useLanguage()
  const tr = t.ratingTool

  const [values, setValues] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleChange = (field: string, val: string) => {
    setValues(prev => ({ ...prev, [field]: val }))
  }

  const handleReset = () => {
    setValues({})
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const payload: Record<string, number | null> = {}
    for (const [k, v] of Object.entries(values)) {
      const num = parseFloat(v)
      payload[k] = isNaN(num) ? null : num
    }

    try {
      const res = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: PredictResult = await res.json()
      setResult(data)
    } catch {
      setError(tr.errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const allFields = FIELD_GROUPS.flatMap(g => g.fields)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
      {/* ── Form panel ─────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-white/8 bg-surface-200/60 backdrop-blur-sm shadow-card overflow-hidden">
        <div className="px-6 py-5 border-b border-white/8">
          <h2 className="text-lg font-semibold text-white">{tr.formTitle}</h2>
          <p className="text-slate-400 text-xs mt-1">{tr.missingNote}</p>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {FIELD_GROUPS.map(group => (
            <div key={group.key}>
              <h3 className="text-xs font-semibold uppercase tracking-widest text-primary-400 mb-3">
                {tr[group.key as keyof typeof tr] as string}
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {group.fields.map(field => (
                  <div key={field}>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      {tr.fields[field as FieldKey]}
                      <span className="ml-1 text-slate-500 font-normal">({tr.optional})</span>
                    </label>
                    <input
                      type="number"
                      step="any"
                      placeholder={tr.placeholders[field as FieldKey]}
                      value={values[field] || ''}
                      onChange={e => handleChange(field, e.target.value)}
                      className="w-full px-3 py-2.5 rounded-xl bg-surface-300/80 border border-white/10
                                 text-white placeholder-slate-500 text-sm focus:outline-none
                                 focus:border-primary-500/60 focus:ring-1 focus:ring-primary-500/30
                                 transition-all"
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              type="submit"
              disabled={loading || allFields.every(f => !values[f])}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl
                         bg-primary-600 hover:bg-primary-500 disabled:opacity-40 disabled:cursor-not-allowed
                         text-white font-semibold text-sm transition-all shadow-glow-primary"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {tr.loadingMsg}
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  {tr.submitBtn}
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-3 rounded-xl border border-white/10 text-slate-400 hover:text-white hover:border-white/20 transition-all text-sm"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>

      {/* ── Result panel ───────────────────────────────────────────── */}
      <div className="rounded-2xl border border-white/8 bg-surface-200/60 backdrop-blur-sm shadow-card min-h-[400px] flex flex-col">
        <div className="px-6 py-5 border-b border-white/8">
          <h2 className="text-lg font-semibold text-white">{tr.resultTitle}</h2>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center p-8">
          {/* Idle state */}
          {!loading && !result && !error && (
            <div className="text-center">
              <div className="w-20 h-20 rounded-full border-2 border-dashed border-white/15 flex items-center justify-center mx-auto mb-4">
                <div className="w-10 h-10 rounded-full bg-primary-500/10 flex items-center justify-center">
                  <Send className="w-5 h-5 text-primary-400" />
                </div>
              </div>
              <p className="text-slate-400 text-sm">{tr.subtitle}</p>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="text-center">
              <div className="w-16 h-16 rounded-full border-2 border-primary-500/30 border-t-primary-400 animate-spin mx-auto mb-4" />
              <p className="text-slate-400 text-sm animate-pulse2">{tr.loadingMsg}</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-3 bg-distressed/10 border border-distressed/30 rounded-xl p-4 w-full">
              <AlertCircle className="w-5 h-5 text-distressed shrink-0 mt-0.5" />
              <p className="text-distressed-light text-sm">{error}</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="w-full space-y-6 animate-fade-in-up">
              {/* Main rating */}
              <div className="text-center">
                <RatingBadge rating={result.rating} size="lg" />
                <p className="mt-2 text-slate-300 text-sm font-medium">
                  {lang === 'vi' ? result.label_vi : result.label_en}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  {tr.confidence}: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>

              {/* Risk score */}
              <div className="bg-surface-300/60 rounded-xl p-4">
                <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>{tr.riskScore}</span>
                  <span className="font-semibold text-white">{result.risk_score.toFixed(0)} / 100</span>
                </div>
                <div className="h-2 rounded-full bg-surface-100/50 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${result.risk_score}%`,
                      backgroundColor: result.color,
                      boxShadow: `0 0 8px ${result.color}`,
                    }}
                  />
                </div>
              </div>

              {/* Probabilities */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
                  {tr.probTitle}
                </h3>
                <ProbabilityBars probabilities={result.probabilities} />
              </div>

              {/* Interpretation */}
              <div className="bg-surface-300/60 rounded-xl p-4">
                <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
                  {tr.interpretTitle}
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {lang === 'vi' ? result.interpretation_vi : result.interpretation_en}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
