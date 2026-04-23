'use client'

import { useState, useEffect, useCallback } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import RatingBadge from './RatingBadge'
import { ChevronLeft, ChevronRight, Search, Filter } from 'lucide-react'
import { API_BASE_URL } from '@/lib/config'

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

  const fetchData = useCallback(async () => {
    setLoading(true)
    const params = new URLSearchParams({
      page: String(page),
      per_page: '15',
      ...(sector ? { sector } : {}),
      ...(rating ? { rating } : {}),
      ...(search ? { search } : {}),
    })
    try {
      const res = await fetch(`${API_BASE_URL}/api/companies?${params}`)
      if (!res.ok) throw new Error('Network response was not ok');
      const json: PageResult = await res.json()
      setData(json)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [page, sector, rating, search])

  useEffect(() => { fetchData() }, [fetchData])

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
          <div className="h-3 animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
        </td>
      ))}
    </tr>
  )

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-col gap-3 lg:flex-row">
        <form onSubmit={handleSearch} className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <label htmlFor="company-search" className="sr-only">{tr.searchPlaceholder}</label>
          <input
            id="company-search"
            type="text"
            placeholder={tr.searchPlaceholder}
            value={searchInput}
            onChange={e => setSearchInput(e.target.value)}
            className="w-full rounded-xl border border-gray-300 bg-white py-3 pl-10 pr-4 text-sm text-gray-800 placeholder:text-gray-400 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-500/10 dark:border-gray-700 dark:bg-gray-900 dark:text-white/90 min-h-[44px]"
          />
          <button
            type="submit"
            aria-label={tr.searchPlaceholder}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg bg-brand-500 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-brand-600 min-h-[36px]"
          >
            {tr.searchBtn}
          </button>
        </form>

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:flex">
          <div className="relative min-w-[170px]">
            <Filter className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" />
            <select
              value={sector}
              onChange={e => { setSector(e.target.value); handleFilterChange() }}
              aria-label={tr.filterSector}
              className="w-full cursor-pointer appearance-none rounded-xl border border-gray-300 bg-white py-3 pl-9 pr-8 text-sm text-gray-700 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-500/10 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 min-h-[44px]"
            >
              <option value="">{tr.filterSector}</option>
              {sectors.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <select
            value={rating}
            onChange={e => { setRating(e.target.value); handleFilterChange() }}
            aria-label={tr.filterRating}
            className="w-full min-w-[140px] cursor-pointer appearance-none rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm text-gray-700 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-500/10 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 min-h-[44px]"
          >
            <option value="">{tr.filterRating}</option>
            <option value="IG">IG</option>
            <option value="HY">HY</option>
            <option value="Distressed">Distressed</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
        <div className="overflow-x-auto">
          <table className="min-w-[720px] w-full text-sm" aria-busy={loading}>
            <caption className="sr-only">{tr.title}</caption>
            <thead>
              <tr className="border-b border-gray-200 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:border-gray-800 dark:text-gray-400">
                <th className="px-3 py-3 text-left sm:px-4">{tr.colCompany}</th>
                <th className="px-3 py-3 text-left sm:px-4">{tr.colTicker}</th>
                <th className="hidden px-3 py-3 text-left md:table-cell sm:px-4">{tr.colSector}</th>
                <th className="px-3 py-3 text-left sm:px-4">{tr.colRating}</th>
                <th className="hidden px-3 py-3 text-left lg:table-cell sm:px-4">{tr.colDate}</th>
                <th className="hidden px-3 py-3 text-left lg:table-cell sm:px-4">{tr.colAgency}</th>
                <th className="hidden px-3 py-3 text-left xl:table-cell sm:px-4">{tr.colSource}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {loading ? (
                Array(8).fill(0).map((_, i) => <Skeleton key={i} />)
              ) : !data || data.data.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-gray-500 dark:text-gray-400">{tr.noResults}</td>
                </tr>
              ) : (
                data.data.map((row, i) => (
                  <tr key={i} className="group transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <td className="max-w-[180px] truncate px-3 py-3 font-medium text-gray-800 dark:text-white/90 sm:px-4">
                      {row.company_name || '-'}
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-brand-500 dark:text-brand-300 sm:px-4">
                      {row.ticker || '-'}
                    </td>
                    <td className="hidden px-3 py-3 text-gray-600 dark:text-gray-300 md:table-cell sm:px-4">{row.sector || '-'}</td>
                    <td className="px-3 py-3 sm:px-4">
                      {row.rating_detail
                        ? <RatingBadge rating={row.rating_detail} size="sm" />
                        : <span className="text-gray-500 dark:text-gray-400">-</span>
                      }
                    </td>
                    <td className="hidden px-3 py-3 font-mono text-xs text-gray-500 dark:text-gray-400 lg:table-cell sm:px-4">{row.rating_date || '-'}</td>
                    <td className="hidden px-3 py-3 text-gray-600 dark:text-gray-300 lg:table-cell sm:px-4">{row.rating_agency || '-'}</td>
                    <td className="hidden px-3 py-3 text-xs text-gray-500 dark:text-gray-400 xl:table-cell sm:px-4">{row.source || '-'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {data && data.total > 0 && (
        <div className="flex flex-col gap-3 text-sm text-gray-600 sm:flex-row sm:items-center sm:justify-between dark:text-gray-300">
          <span>
            {tr.showing} {((data.page - 1) * data.per_page) + 1} - {Math.min(data.page * data.per_page, data.total)} {tr.of} {data.total.toLocaleString()} {tr.records}
          </span>
          <div className="flex items-center gap-2 self-end sm:self-auto">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={data.page === 1}
              aria-label={tr.prev}
              className="flex items-center gap-1 rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-100 disabled:opacity-40 dark:border-gray-700 dark:hover:bg-gray-800 min-h-[44px] sm:min-h-0 sm:px-3 sm:py-1.5"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              {tr.prev}
            </button>
            <span className="flex items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-800 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 min-h-[44px] sm:min-h-0 sm:px-3 sm:py-1.5">
              {data.page} / {data.pages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(data.pages, p + 1))}
              disabled={data.page === data.pages}
              aria-label={tr.next}
              className="flex items-center gap-1 rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-100 disabled:opacity-40 dark:border-gray-700 dark:hover:bg-gray-800 min-h-[44px] sm:min-h-0 sm:px-3 sm:py-1.5"
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
