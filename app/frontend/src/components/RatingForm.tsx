'use client'

import { useRef, useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import { Send, RotateCcw, AlertCircle, Upload, BarChart3 } from 'lucide-react'
import ProbabilityBars from './ProbabilityBars'
import RatingBadge from './RatingBadge'
import SpNotchBadge from '@/components/SpNotchBadge'
import { API_BASE_URL } from '@/lib/config'
import { apiFetch } from '@/lib/api'

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

const ALL_FIELDS = FIELD_GROUPS.flatMap(g => g.fields)

interface PredictResult {
  model?: string
  rating: string
  probabilities: Record<string, number>
  confidence: number
  risk_level: string
  risk_score: number
  interpretation: string
  label: string
  color: string
  sector_resolved?: string
  previous_rating?: string
}

interface ExplainResult {
  provider: string
  model: string
  explanation: string
  xai_used?: boolean
}

interface SpContext {
  rating_class: string
  indicative_notch: string
  indicative_range: string
  range_low: string
  range_high: string
  confidence_band: string
  risk_band: string
  sp_scale: string[]
  disclaimer: string
  migration_note: string
}

type FieldKey = typeof FIELD_GROUPS[number]['fields'][number]

type ShapLikeRule = {
  neutral: number
  riskyWhen: 'higher' | 'lower'
  label: string
}

const SHAP_LIKE_RULES: Record<FieldKey, ShapLikeRule> = {
  current_ratio: { neutral: 1.5, riskyWhen: 'lower', label: 'Current ratio' },
  debt_equity_ratio: { neutral: 1.0, riskyWhen: 'higher', label: 'Debt to equity' },
  gross_profit_margin: { neutral: 0.25, riskyWhen: 'lower', label: 'Gross margin' },
  operating_profit_margin: { neutral: 0.12, riskyWhen: 'lower', label: 'Operating margin' },
  ebit_margin: { neutral: 0.1, riskyWhen: 'lower', label: 'EBIT margin' },
  pretax_profit_margin: { neutral: 0.08, riskyWhen: 'lower', label: 'Pre-tax margin' },
  net_profit_margin: { neutral: 0.06, riskyWhen: 'lower', label: 'Net margin' },
  asset_turnover: { neutral: 0.8, riskyWhen: 'lower', label: 'Asset turnover' },
  roe: { neutral: 0.12, riskyWhen: 'lower', label: 'ROE' },
  roa: { neutral: 0.05, riskyWhen: 'lower', label: 'ROA' },
  operating_cashflow_ps: { neutral: 0.0, riskyWhen: 'lower', label: 'Operating cashflow/share' },
  free_cashflow_ps: { neutral: 0.0, riskyWhen: 'lower', label: 'Free cashflow/share' },
}

type CsvRow = Partial<Record<FieldKey, string>> & {
  __ticker?: string
  __company?: string
  __sector?: string
}

type AppliedCsvMeta = {
  ticker?: string
  companyName?: string
  sector?: string
}

const normalizeHeader = (header: string) =>
  header
    .replace(/^\uFEFF/, '')
    .trim()
    .toLowerCase()
    .replace(/[%()]/g, '')
    .replace(/[^a-z0-9]+/g, '')

const FIELD_BY_NORMALIZED = ALL_FIELDS.reduce((acc, field) => {
  acc[normalizeHeader(field)] = field
  return acc
}, {} as Record<string, FieldKey>)

const HEADER_ALIAS_TO_FIELD: Record<string, FieldKey> = {
  currentratio: 'current_ratio',
  debttoequity: 'debt_equity_ratio',
  debtequity: 'debt_equity_ratio',
  debtequityratio: 'debt_equity_ratio',
  grossprofitmargin: 'gross_profit_margin',
  gpm: 'gross_profit_margin',
  operatingprofitmargin: 'operating_profit_margin',
  opm: 'operating_profit_margin',
  ebitmargin: 'ebit_margin',
  pretaxprofitmargin: 'pretax_profit_margin',
  netprofitmargin: 'net_profit_margin',
  assetturnover: 'asset_turnover',
  roe: 'roe',
  returnonequity: 'roe',
  roa: 'roa',
  returnonassets: 'roa',
  operatingcashflowps: 'operating_cashflow_ps',
  operatingcashflowpershare: 'operating_cashflow_ps',
  freecashflowps: 'free_cashflow_ps',
  freecashflowpershare: 'free_cashflow_ps',
}

const TICKER_HEADER_ALIASES = new Set(['ticker', 'symbol', 'stockcode', 'stockticker', 'mack'])
const COMPANY_HEADER_ALIASES = new Set(['company', 'companyname', 'name', 'issuer'])
const SECTOR_HEADER_ALIASES = new Set(['sector', 'industry', 'linhvuc', 'nganh'])

const resolveFieldFromHeader = (header: string): FieldKey | null => {
  const normalized = normalizeHeader(header)
  return FIELD_BY_NORMALIZED[normalized] || HEADER_ALIAS_TO_FIELD[normalized] || null
}

const detectDelimiter = (content: string) => {
  const firstLine = content.split(/\r?\n/, 1)[0] || ''
  const commaCount = (firstLine.match(/,/g) || []).length
  const semicolonCount = (firstLine.match(/;/g) || []).length
  return semicolonCount > commaCount ? ';' : ','
}

const parseCsvMatrix = (content: string, delimiter: string): string[][] => {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let inQuotes = false

  for (let i = 0; i < content.length; i += 1) {
    const char = content[i]

    if (char === '"') {
      if (inQuotes && content[i + 1] === '"') {
        cell += '"'
        i += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (!inQuotes && char === delimiter) {
      row.push(cell.trim())
      cell = ''
      continue
    }

    if (!inQuotes && (char === '\n' || char === '\r')) {
      if (char === '\r' && content[i + 1] === '\n') i += 1
      row.push(cell.trim())
      rows.push(row)
      row = []
      cell = ''
      continue
    }

    cell += char
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell.trim())
    rows.push(row)
  }

  return rows
}

const toClampedPercent = (value: number) => {
  const finite = Number.isFinite(value) ? value : 0
  return Math.max(0, Math.min(100, finite))
}

const toSafeNumber = (value: number) => (Number.isFinite(value) ? value : 0)

const getRiskBgClass = (riskLevel: string, rating: string) => {
  const lvl = (riskLevel || '').toLowerCase()
  if (lvl.includes('low') || rating === 'IG') return 'bg-ig'
  if (lvl.includes('high') || lvl.includes('distressed') || rating === 'Distressed') return 'bg-distressed'
  return 'bg-hy'
}

export default function RatingForm() {
  const { t, lang } = useLanguage()
  const tr = t.ratingTool

  const buildShapLikeDrivers = (payload: Record<string, number | null>) => {
    const drivers = ALL_FIELDS
      .map((field) => {
        const value = payload[field]
        if (value === null || !Number.isFinite(value)) return null

        const rule = SHAP_LIKE_RULES[field]
        const denominator = Math.abs(rule.neutral) > 1e-6 ? Math.abs(rule.neutral) : 1
        const deviationRatio = Math.abs(value - rule.neutral) / denominator
        const increasesRisk = rule.riskyWhen === 'higher' ? value > rule.neutral : value < rule.neutral
        const signedScore = Math.min(2.5, deviationRatio) * (increasesRisk ? 1 : -1)

        return {
          feature: field,
          label: rule.label,
          value,
          neutral_reference: rule.neutral,
          risky_when: rule.riskyWhen,
          impact_direction: increasesRisk ? 'increases_risk' : 'reduces_risk',
          shap_proxy_score: Number(signedScore.toFixed(4)),
          abs_impact_strength: Number(Math.abs(signedScore).toFixed(4)),
        }
      })
      .filter((item): item is NonNullable<typeof item> => item !== null)
      .sort((a, b) => b.abs_impact_strength - a.abs_impact_strength)

    return drivers.slice(0, 6)
  }

  const buildXaiContext = (prediction: PredictResult, payload: Record<string, number | null>) => {
    const probabilityRanking = Object.entries(prediction.probabilities || {})
      .map(([label, probability]) => ({ label, probability }))
      .sort((a, b) => b.probability - a.probability)

    const missingFeatures = ALL_FIELDS.filter(field => payload[field] === null)
    const providedFeatures = ALL_FIELDS.filter(field => payload[field] !== null)
    const shapStyleTopDrivers = buildShapLikeDrivers(payload)

    return {
      source: 'frontend-synthesized-xai',
      xai_mode: 'rule-based-shap-proxy',
      xai_note: 'SHAP-style proxy from directional distance to neutral financial anchors (not exact Shapley values).',
      top_prediction: {
        rating: prediction.rating,
        confidence: prediction.confidence,
        risk_level: prediction.risk_level,
        risk_score: prediction.risk_score,
      },
      probability_ranking: probabilityRanking,
      shap_style_top_drivers: shapStyleTopDrivers,
      feature_coverage: {
        provided_count: providedFeatures.length,
        missing_count: missingFeatures.length,
        missing_features: missingFeatures,
      },
      model_interpretation: {
        vi: lang === 'vi' ? prediction.interpretation : '',
        en: lang === 'en' ? prediction.interpretation : '',
      },
    }
  }

  const buildTlstmExplainInput = (prediction: PredictResult) => ({
    model: prediction.model || 'TLSTMFuzzy',
    rating: prediction.rating,
    probabilities: prediction.probabilities,
    confidence: prediction.confidence,
    risk_level: prediction.risk_level,
    risk_score: prediction.risk_score,
    label_en: prediction.label,
    label_vi: prediction.label,
    interpretation_en: prediction.interpretation,
    interpretation_vi: prediction.interpretation,
    sector_resolved: prediction.sector_resolved,
    previous_rating: prediction.previous_rating,
  })

  const streamExplain = async (payload: Record<string, number | null>, prediction: PredictResult) => {
    setExplainLoading(true)
    setExplainText('')
    setExplainError(null)
    setExplainModel(null)
    setSpContext(null)

    const xaiContext = buildXaiContext(prediction, payload)
    const tlstmPrediction = buildTlstmExplainInput(prediction)

    const decoder = new TextDecoder()
    let buffer = ''
    let firstTokenSeen = false

    try {
      const explainRes = await fetch(`${API_BASE_URL}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: payload,
          prediction,
          tlstm_prediction: tlstmPrediction,
          lang,
          xai_context: xaiContext,
          stream: true,
        }),
      })

      if (!explainRes.ok) {
        let detail = `HTTP ${explainRes.status}`
        try {
          const errBody = await explainRes.json()
          detail = errBody?.detail || detail
        } catch {
          // Keep fallback HTTP status text when body is not JSON.
        }
        throw new Error(detail)
      }

      const reader = explainRes.body?.getReader()
      if (!reader) {
        throw new Error('Explain stream is unavailable.')
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const rawLine of lines) {
          const line = rawLine.trimEnd()
          if (!line.startsWith('data: ')) continue

          const eventText = line.slice('data: '.length).trim()
          if (!eventText) continue

          let event: unknown
          try {
            event = JSON.parse(eventText) as unknown
          } catch {
            continue
          }

          if (typeof event === 'object' && event !== null && 'token' in event) {
            const token = (event as { token?: unknown }).token
            if (typeof token === 'string' && token.length > 0) {
              if (!firstTokenSeen) {
                firstTokenSeen = true
                setExplainLoading(false)
              }
              setExplainText(prev => (prev || '') + token)
            }
            continue
          }

          if (typeof event === 'object' && event !== null && 'done' in event) {
            const doneFlag = (event as { done?: unknown }).done
            const ctx = (event as { sp_context?: unknown }).sp_context
            if (doneFlag === true && typeof ctx === 'object' && ctx !== null) {
              setSpContext(ctx as SpContext)
            }
            setExplainLoading(false)
            continue
          }

          if (typeof event === 'object' && event !== null && 'error' in event) {
            const errMsg = (event as { error?: unknown }).error
            if (typeof errMsg === 'string' && errMsg.length > 0) {
              setExplainError(formatExplainError(errMsg))
            } else {
              setExplainError(formatExplainError(tr.explainErrorMsg))
            }
            setExplainLoading(false)
          }
        }
      }

      buffer += decoder.decode()
      const tailLines = buffer.split('\n')
      for (const rawLine of tailLines) {
        const line = rawLine.trimEnd()
        if (!line.startsWith('data: ')) continue
        const eventText = line.slice('data: '.length).trim()
        if (!eventText) continue

        try {
          const event = JSON.parse(eventText) as unknown
          if (typeof event === 'object' && event !== null && 'done' in event) {
            const doneFlag = (event as { done?: unknown }).done
            const ctx = (event as { sp_context?: unknown }).sp_context
            if (doneFlag === true && typeof ctx === 'object' && ctx !== null) {
              setSpContext(ctx as SpContext)
            }
          }
        } catch {
          // Ignore tail parse errors.
        }
      }
    } catch (explainErr) {
      const rawMessage = explainErr instanceof Error ? explainErr.message : tr.explainErrorMsg
      setExplainError(formatExplainError(rawMessage))
    } finally {
      setExplainLoading(false)
    }
  }

  const formatBackendError = (message: string) => {
    const msg = message.toLowerCase()
    if (msg.includes('failed to fetch') || msg.includes('networkerror') || msg.includes('load failed')) {
      return lang === 'vi'
        ? 'Khong ket noi duoc backend. Hay khoi dong backend tai http://localhost:8000 va thu lai.'
        : 'Cannot connect to backend. Please start backend at http://localhost:8000 and try again.'
    }
    if (msg.includes('still starting up')) {
      return lang === 'vi'
        ? 'Backend dang khoi dong model. Vui long doi vai giay roi thu lai.'
        : 'Backend is still warming up models. Please wait a few seconds and try again.'
    }
    if (msg.includes('lightgbm model unavailable')) {
      return lang === 'vi'
        ? 'Model LightGBM chua san sang. Kiem tra log backend hoac dung endpoint TLSTM.'
        : 'LightGBM model is unavailable. Check backend logs or use the TLSTM endpoint.'
    }
    if (msg.includes('tlstmfuzzy model is not available') || msg.includes('tlstm model is not available')) {
      return lang === 'vi'
        ? 'Model TLSTMFuzzy chua san sang. Hay kiem tra log backend va thu lai.'
        : 'TLSTMFuzzy model is not available. Please check backend logs and try again.'
    }
    if (msg.includes('prediction error:')) {
      return message.replace(/^prediction error:\s*/i, '')
    }
    return message
  }

  const formatExplainError = (message: string) => {
    const msg = message.toLowerCase()
    if (msg.includes('failed to fetch') || msg.includes('networkerror') || msg.includes('load failed')) {
      return lang === 'vi'
        ? 'Khong ket noi duoc backend explain. Hay kiem tra backend va thu lai.'
        : 'Cannot connect to explain backend. Please check backend and try again.'
    }
    if (msg.includes('api key is missing') || msg.includes('gemini api key is missing') || msg.includes('google api key is missing')) {
      return lang === 'vi'
        ? 'Backend chua cau hinh GEMINI_API_KEY. Hay them bien moi truong va khoi dong lai backend.'
        : 'Backend is missing GEMINI_API_KEY. Please set the environment variable and restart backend.'
    }
    if (msg.includes('model') && (msg.includes('not found') || msg.includes('not available'))) {
      return lang === 'vi'
        ? 'Model Gemini hien tai khong ho tro. Dat GEMINI_MODEL phu hop va khoi dong lai backend.'
        : 'The configured Gemini model is not supported. Set a valid GEMINI_MODEL and restart backend.'
    }
    if (msg.includes('quota exceeded') || msg.includes('resource_exhausted')) {
      return lang === 'vi'
        ? 'Da vuot han muc Gemini (quota). Vui long kiem tra billing/quota va thu lai sau.'
        : 'Gemini quota is exceeded. Please check billing/quota and try again later.'
    }
    return message || tr.explainErrorMsg
  }

  const csvInputRef = useRef<HTMLInputElement | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [csvRows, setCsvRows] = useState<CsvRow[]>([])
  const [csvSelectedIndex, setCsvSelectedIndex] = useState(0)
  const [csvFileName, setCsvFileName] = useState('')
  const [csvStatus, setCsvStatus] = useState<string | null>(null)
  const [csvError, setCsvError] = useState<string | null>(null)
  const [appliedCsvMeta, setAppliedCsvMeta] = useState<AppliedCsvMeta | null>(null)
  const [explainLoading, setExplainLoading] = useState(false)
  const [explainText, setExplainText] = useState<string | null>(null)
  const [explainError, setExplainError] = useState<string | null>(null)
  const [explainModel, setExplainModel] = useState<string | null>(null)
  const [spContext, setSpContext] = useState<SpContext | null>(null)

  const handleChange = (field: string, val: string) => {
    if (appliedCsvMeta) {
      setAppliedCsvMeta(null)
    }
    setValues(prev => ({ ...prev, [field]: val }))
  }

  const handleReset = () => {
    setValues({})
    setResult(null)
    setError(null)
    setCsvRows([])
    setCsvSelectedIndex(0)
    setCsvFileName('')
    setCsvStatus(null)
    setCsvError(null)
    setAppliedCsvMeta(null)
    setExplainLoading(false)
    setExplainText(null)
    setExplainError(null)
    setExplainModel(null)
    setSpContext(null)
    setSpContext(null)
    if (csvInputRef.current) {
      csvInputRef.current.value = ''
    }
  }

  const buildCsvRowLabel = (row: CsvRow, index: number) => {
    if (row.__ticker && row.__company) {
      return `${index + 1}. ${row.__ticker} - ${row.__company}`
    }
    if (row.__ticker) {
      return `${index + 1}. ${row.__ticker}`
    }
    if (row.__company) {
      return `${index + 1}. ${row.__company}`
    }
    return `${tr.csvRowFallback} ${index + 1}`
  }

  const applyCsvRowToForm = () => {
    const row = csvRows[csvSelectedIndex]
    if (!row) return

    const nextValues: Record<string, string> = {}
    for (const field of ALL_FIELDS) {
      nextValues[field] = row[field] ?? ''
    }
    setValues(nextValues)
    setAppliedCsvMeta({
      ticker: row.__ticker?.trim() || undefined,
      companyName: row.__company?.trim() || undefined,
      sector: row.__sector?.trim() || undefined,
    })
    setResult(null)
    setError(null)
    setCsvStatus(tr.csvAutofillSuccess)
  }

  const handleCsvImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setCsvFileName(file.name)
    setCsvStatus(null)
    setCsvError(null)

    try {
      const rawText = await file.text()
      const content = rawText.replace(/^\uFEFF/, '')
      const delimiter = detectDelimiter(content)
      const matrix = parseCsvMatrix(content, delimiter)

      if (matrix.length < 2) {
        setCsvRows([])
        setCsvError(tr.csvNoDataRows)
        return
      }

      const headers = matrix[0]
      const fieldByIndex = headers.map(resolveFieldFromHeader)
      const mappedColumns = fieldByIndex.filter(Boolean).length

      if (mappedColumns === 0) {
        setCsvRows([])
        setCsvError(tr.csvNoMappedColumns)
        return
      }

      let tickerIndex = -1
      let companyIndex = -1
      let sectorIndex = -1
      headers.forEach((header, idx) => {
        const normalized = normalizeHeader(header)
        if (tickerIndex === -1 && TICKER_HEADER_ALIASES.has(normalized)) tickerIndex = idx
        if (companyIndex === -1 && COMPANY_HEADER_ALIASES.has(normalized)) companyIndex = idx
        if (sectorIndex === -1 && SECTOR_HEADER_ALIASES.has(normalized)) sectorIndex = idx
      })

      const parsedRows: CsvRow[] = matrix
        .slice(1)
        .filter(row => row.some(cell => cell.trim() !== ''))
        .map(rawRow => {
          const parsedRow: CsvRow = {}
          fieldByIndex.forEach((field, colIdx) => {
            if (!field) return
            parsedRow[field] = (rawRow[colIdx] || '').trim()
          })
          if (tickerIndex >= 0) parsedRow.__ticker = (rawRow[tickerIndex] || '').trim()
          if (companyIndex >= 0) parsedRow.__company = (rawRow[companyIndex] || '').trim()
          if (sectorIndex >= 0) parsedRow.__sector = (rawRow[sectorIndex] || '').trim()
          return parsedRow
        })
        .filter(row => ALL_FIELDS.some(field => (row[field] || '').trim() !== ''))

      if (parsedRows.length === 0) {
        setCsvRows([])
        setCsvError(tr.csvNoDataRows)
        return
      }

      setCsvRows(parsedRows)
      setCsvSelectedIndex(0)
      setCsvStatus(`${tr.csvParseSuccess} ${parsedRows.length}. ${tr.csvMappedColumns}: ${mappedColumns}/${ALL_FIELDS.length}. ${tr.csvMappedByHeader}`)
    } catch {
      setCsvRows([])
      setCsvError(tr.csvParseError)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    setExplainLoading(false)
    setExplainText(null)
    setExplainError(null)
    setExplainModel(null)

    const payload: Record<string, number | null> = {}
    for (const field of ALL_FIELDS) {
      const rawValue = values[field]
      if (rawValue === undefined || rawValue === '') {
        payload[field] = null
        continue
      }
      const num = parseFloat(rawValue)
      payload[field] = isNaN(num) ? null : num
    }

    try {
      const data = await apiFetch<PredictResult>('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }, lang)
      setResult(data)
      setLoading(false)

      void fetch(`${API_BASE_URL}/api/history`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prediction: data,
            features: payload,
            ticker: appliedCsvMeta?.ticker,
            company_name: appliedCsvMeta?.companyName,
            sector: appliedCsvMeta?.sector,
            source: 'VN-Rating Analyze',
          }),
        })
        .catch(() => {
        // Keep UX non-blocking: prediction already succeeded even if history save fails.
        })

      void streamExplain(payload, data)
    } catch (err) {
      const rawMessage = err instanceof Error ? err.message : tr.errorMsg
      setError(formatBackendError(rawMessage || tr.errorMsg))
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      const isSubmittable = !loading && !ALL_FIELDS.every(f => !values[f])
      if (isSubmittable) {
        handleSubmit(e as unknown as React.FormEvent)
      }
    }
  }

  return (
    <div className={`flex flex-col ${result || loading || error ? 'flex-col-reverse' : ''} lg:grid lg:grid-cols-2 items-start gap-12 lg:gap-16`}>
      {/* ── Form panel ─────────────────────────────────────────────── */}
      <section className="flex flex-col gap-6 w-full min-w-0">
        <header className="border-b border-gray-900 pb-4 dark:border-white flex justify-between items-end">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">{tr.formTitle}</h2>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{tr.missingNote}</p>
          </div>
          <div className="hidden text-xs text-gray-400 dark:text-gray-500 sm:flex items-center gap-1 pb-1">
            <kbd className="font-sans font-semibold">⌘</kbd>
            <span className="text-[10px] uppercase tracking-widest leading-none mt-0.5">Enter</span>
          </div>
        </header>

        <form onSubmit={handleSubmit} onKeyDown={handleKeyDown} className="space-y-8 pt-2">
          <div className="border-b border-gray-200 pb-8 dark:border-gray-800">
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-900 dark:text-white">{tr.csvTitle}</h3>
                <p className="mt-2 text-xs leading-relaxed text-gray-500 dark:text-gray-400 max-w-sm">
                  {tr.csvHint} {tr.csvScaleHint}
                </p>
              </div>
              <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-none border border-gray-900 bg-gray-900 px-4 py-2.5 text-xs font-bold uppercase tracking-widest text-white transition-colors hover:bg-gray-800 dark:border-white dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100 shrink-0">
                <Upload className="w-4 h-4" />
                {tr.csvChooseFile}
                <input
                  ref={csvInputRef}
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleCsvImport}
                  className="hidden"
                />
              </label>
            </div>

            {csvFileName && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {tr.csvFileLabel}: <span className="text-gray-700 dark:text-gray-200">{csvFileName}</span>
              </p>
            )}

            {csvStatus && (
              <p className="text-xs text-ig-dark dark:text-ig" role="status" aria-live="polite">{csvStatus}</p>
            )}

            {csvError && (
              <p className="text-xs text-distressed-dark dark:text-distressed" role="alert">{csvError}</p>
            )}

            {csvRows.length > 0 && (
              <div className="mt-6 flex flex-col sm:flex-row gap-4 w-full max-w-full overflow-hidden">
                <select
                  value={csvSelectedIndex}
                  onChange={e => setCsvSelectedIndex(Number(e.target.value))}
                  aria-label="CSV row selection"
                  className="flex-1 min-w-0 rounded-none border-0 border-b-2 border-gray-200 bg-transparent py-2 pl-0 pr-8 text-sm font-semibold text-gray-900 focus:border-brand-500 focus:ring-0 dark:border-gray-800 dark:text-white truncate"
                >
                  {csvRows.map((row, idx) => (
                    <option key={`${row.__ticker || 'row'}-${idx}`} value={idx}>
                      {buildCsvRowLabel(row, idx)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={applyCsvRowToForm}
                  className="shrink-0 rounded-none border border-gray-900 px-5 py-2.5 text-xs font-bold uppercase tracking-widest text-gray-900 transition-colors hover:bg-gray-100 dark:border-white dark:text-white dark:hover:bg-white/10 whitespace-nowrap"
                >
                  {tr.csvApplyRow}
                </button>
              </div>
            )}
          </div>

          {FIELD_GROUPS.map(group => (
            <div key={group.key} className="space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-widest text-brand-600 dark:text-brand-400">
                {tr[group.key as keyof typeof tr] as string}
              </h3>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {group.fields.map(field => {
                  const inputId = `rating-field-${field}`

                  return (
                  <div key={field}>
                    <label htmlFor={inputId} className="mb-1 block text-xs font-medium text-gray-700 dark:text-gray-200">
                      {tr.fields[field as FieldKey]}
                      <span className="ml-1 font-normal text-gray-500 dark:text-gray-400">({tr.optional})</span>
                    </label>
                    <input
                      id={inputId}
                      type="number"
                      step="any"
                      inputMode="decimal"
                      placeholder={tr.placeholders[field as FieldKey]}
                      value={values[field] || ''}
                      onChange={e => handleChange(field, e.target.value)}
                      className="w-full rounded-none border-0 border-b-2 border-gray-200 bg-transparent px-0 py-2 text-sm font-medium text-gray-900 placeholder:font-normal placeholder:text-gray-400 focus:border-brand-500 focus:ring-0 dark:border-gray-800 dark:text-white"
                    />
                  </div>
                )})}
              </div>
            </div>
          ))}

          {/* Actions */}
          <div className="flex flex-col gap-3 pt-6 sm:flex-row">
            <button
              type="submit"
              disabled={loading || ALL_FIELDS.every(f => !values[f])}
              className="flex flex-1 items-center justify-center gap-2 rounded-none bg-brand-600 px-6 py-4 text-xs font-bold uppercase tracking-widest text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
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
              aria-label={tr.resetBtn}
              className="flex items-center justify-center rounded-none border border-gray-300 px-6 py-4 text-gray-500 transition-colors hover:bg-gray-100 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800 sm:w-auto"
            >
              <span className="inline-flex items-center gap-2">
                <RotateCcw className="w-4 h-4" />
                <span className="sr-only text-xs font-bold uppercase tracking-widest sm:not-sr-only">{tr.resetBtn}</span>
              </span>
            </button>
          </div>
        </form>
      </section>

      {/* ── Result panel ───────────────────────────────────────────── */}
      <section className="flex min-h-0 flex-col gap-6 lg:min-h-[400px]">
        <header className="border-b border-gray-900 pb-4 dark:border-white">
          <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white font-serif tracking-normal">{tr.resultTitle}</h2>
        </header>

        <div className="flex flex-1 flex-col items-center justify-center py-5">
          {/* Idle state */}
          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center opacity-30 my-auto py-12">
              <BarChart3 className="h-16 w-16 text-gray-900 dark:text-white mb-6 stroke-1" />
              <p className="text-sm font-bold uppercase tracking-widest text-gray-900 dark:text-white">{tr.subtitle}</p>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="text-center" role="status" aria-live="polite">
              <div className="mx-auto mb-4 h-16 w-16 animate-spin rounded-full border-2 border-brand-300/40 border-t-brand-500" />
              <p className="text-sm text-gray-500 dark:text-gray-400">{tr.loadingMsg}</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex w-full items-start gap-3 rounded-none border border-distressed/30 bg-distressed/10 p-4" role="alert">
              <AlertCircle className="w-5 h-5 text-distressed-dark shrink-0 mt-0.5 dark:text-distressed" />
              <p className="text-sm text-distressed-dark dark:text-distressed">{error}</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="w-full space-y-6">
              {/* Main rating */}
              <div className="text-center">
                <RatingBadge rating={result.rating} size="lg" />
                <p className="mt-2 text-sm font-medium text-gray-700 dark:text-gray-200">
                  {result.label}
                </p>
                <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
                  {tr.confidence}: {(toSafeNumber(result.confidence) * 100).toFixed(1)}%
                </p>
              </div>

              {/* Risk score */}
              <div className="border-t border-gray-200 pt-6 dark:border-gray-800">
                <div className="mb-2 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                  <span>{tr.riskScore}</span>
                  <span className="font-semibold text-gray-800 dark:text-white">{toSafeNumber(result.risk_score).toFixed(0)} / 100</span>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-800">
                  <div
                    className={`absolute inset-0 h-full w-full rounded-full transition-transform duration-700 ease-out ${getRiskBgClass(result.risk_level, result.rating)}`}
                    style={{
                      transform: `translateX(-${100 - toClampedPercent(toSafeNumber(result.risk_score))}%)`,
                    }}
                  />
                </div>
              </div>

              {/* Probabilities */}
              <div className="border-t border-gray-200 pt-6 dark:border-gray-800">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                  {tr.probTitle}
                </h3>
                <ProbabilityBars probabilities={result.probabilities} />
              </div>

              {spContext && !explainLoading && (
                <SpNotchBadge spContext={spContext} />
              )}

              {explainLoading && (
                <div className="border-t border-gray-200 pt-6 dark:border-gray-800" role="status" aria-live="polite">
                  <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-brand-500" />
                    <span>{tr.explainLoadingMsg}</span>
                  </div>
                </div>
              )}

              {!explainLoading && explainError && (
                <div className="border-t border-gray-200 pt-6 dark:border-gray-800">
                  <p className="text-sm leading-relaxed text-distressed-dark dark:text-distressed">{explainError}</p>
                </div>
              )}

              {!explainLoading && !explainError && explainText && (
                <div className="border-t border-gray-200 pt-6 dark:border-gray-800">
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                    {tr.interpretTitle}
                  </h3>
                  {explainModel && (
                    <p className="mb-2 text-xs text-gray-500 dark:text-gray-400">{explainModel}</p>
                  )}
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p className="max-h-56 overflow-y-auto whitespace-pre-line text-sm leading-relaxed text-gray-700 dark:text-gray-300 font-serif italic">
                      {explainText}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
