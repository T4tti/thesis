'use client'

import { useState, useEffect } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import RatingBadge from './RatingBadge'
import { ChevronLeft, ChevronRight, Search, Filter } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Company {
  company_name: string
  ticker: string
  sector: string
  rating_detail: string
  rating_date: string
  rating_agency: string
  source: string
}

interface PageResult {
  data: Company[]
  total: number
  page: number
  per_page: number
  pages: number
}

interface Props {
  sectors: string[]
}

export default function DataTable({ sectors }: Props) {
  const { t } = useLanguage()
  const tr = t.reports

  const [data, setData] = useState<PageResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [sector, setSector] = useState('')
  const [rating, setRating] = useState('')
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')

  const fetchData = async () => {
    setLoading(true)
    const params = new URLSearchParams({
      page: String(page),
      per_page: '15',
      ...(sector ? { sector } : {}),
      ...(rating ? { rating } : {}),
      ...(search ? { search } : {}),
    })
    try {
      const res = await fetch(`${API}/api/companies?${params}`)
      const json: PageResult = await res.json()
      setData(json)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [page, sector, rating, search]) // eslint-disable-line

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setSearch(searchInput)
    setPage(1)
  }

  const handleFilterChange = () => setPage(1)

  const Skeleton = () => (
    <tr>
      {Array(7).fill(0).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-3 bg-surface-100/40 rounded animate-shimmer bg-gradient-to-r from-surface-100/20 via-surface-100/40 to-surface-100/20" style={{backgroundSize:'200% 100%'}} />
        </td>
      ))}
    </tr>
  )

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <form onSubmit={handleSearch} className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder={tr.searchPlaceholder}
            value={searchInput}
            onChange={e => setSearchInput(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 rounded-xl bg-surface-200/80 border border-white/10
                       text-white placeholder-slate-500 text-sm focus:outline-none
                       focus:border-primary-500/60 focus:ring-1 focus:ring-primary-500/30 transition-all"
          />
        </form>

        <div className="flex gap-2">
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <select
              value={sector}
              onChange={e => { setSector(e.target.value); handleFilterChange() }}
              className="pl-8 pr-8 py-2.5 rounded-xl bg-surface-200/80 border border-white/10 text-white text-sm
                         focus:outline-none focus:border-primary-500/60 transition-all appearance-none cursor-pointer"
            >
              <option value="">{tr.filterSector}</option>
              {sectors.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <select
            value={rating}
            onChange={e => { setRating(e.target.value); handleFilterChange() }}
            className="px-4 py-2.5 rounded-xl bg-surface-200/80 border border-white/10 text-white text-sm
                       focus:outline-none focus:border-primary-500/60 transition-all appearance-none cursor-pointer"
          >
            <option value="">{tr.filterRating}</option>
            <option value="IG">IG</option>
            <option value="HY">HY</option>
            <option value="Distressed">Distressed</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-2xl border border-white/8 bg-surface-200/50 overflow-hidden shadow-card">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/8 text-xs font-semibold uppercase tracking-wider text-slate-500">
                <th className="px-4 py-3.5 text-left">{tr.colCompany}</th>
                <th className="px-4 py-3.5 text-left">{tr.colTicker}</th>
                <th className="px-4 py-3.5 text-left">{tr.colSector}</th>
                <th className="px-4 py-3.5 text-left">{tr.colRating}</th>
                <th className="px-4 py-3.5 text-left">{tr.colDate}</th>
                <th className="px-4 py-3.5 text-left">{tr.colAgency}</th>
                <th className="px-4 py-3.5 text-left">{tr.colSource}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/4">
              {loading ? (
                Array(8).fill(0).map((_, i) => <Skeleton key={i} />)
              ) : !data || data.data.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-500">{tr.noResults}</td>
                </tr>
              ) : (
                data.data.map((row, i) => (
                  <tr key={i} className="hover:bg-white/3 transition-colors group">
                    <td className="px-4 py-3 font-medium text-white truncate max-w-[180px]">
                      {row.company_name || '—'}
                    </td>
                    <td className="px-4 py-3 font-mono text-primary-300 text-xs">
                      {row.ticker || '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-400">{row.sector || '—'}</td>
                    <td className="px-4 py-3">
                      {row.rating_detail
                        ? <RatingBadge rating={row.rating_detail} size="sm" />
                        : <span className="text-slate-500">—</span>
                      }
                    </td>
                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">{row.rating_date || '—'}</td>
                    <td className="px-4 py-3 text-slate-400">{row.rating_agency || '—'}</td>
                    <td className="px-4 py-3 text-slate-500 text-xs">{row.source || '—'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {data && data.total > 0 && (
        <div className="flex items-center justify-between text-sm text-slate-400">
          <span>
            {tr.showing} {((data.page - 1) * data.per_page) + 1}–{Math.min(data.page * data.per_page, data.total)} {tr.of} {data.total.toLocaleString()} {tr.records}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={data.page === 1}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-white/10 hover:border-white/20 hover:text-white disabled:opacity-30 transition-all"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              {tr.prev}
            </button>
            <span className="px-3 py-1.5 bg-surface-200/60 rounded-lg border border-white/8 text-white">
              {data.page} / {data.pages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(data.pages, p + 1))}
              disabled={data.page === data.pages}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-white/10 hover:border-white/20 hover:text-white disabled:opacity-30 transition-all"
            >
              {tr.next}
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
