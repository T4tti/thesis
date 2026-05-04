'use client'

import Link from 'next/link'
import { useLanguage } from '@/context/LanguageContext'
import RatingBadge from '@/components/RatingBadge'
import { ArrowRight, BarChart2, Database, TrendingUp, Zap, Shield, AlertTriangle } from 'lucide-react'

interface Stats {
  total_records: number
  unique_tickers: number
  unique_sectors: number
  model_cv_f1_weighted: number
}

interface Props {
  stats: Stats | null
}

export default function HomeClient({ stats }: Props) {
  const { t } = useLanguage()
  const h = t.home

  return (
    <div className="space-y-16 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* 1. Hero Section: Editorial & Bold */}
      <section className="flex flex-col gap-10 border-b border-gray-200 pb-12 dark:border-gray-800 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-600 dark:border-brand-500/30 dark:bg-brand-500/10 dark:text-brand-300">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-500"></span>
            </span>
            {h.badge}
          </div>
          <h1 className="text-4xl font-light tracking-tight text-gray-900 dark:text-white md:text-5xl lg:text-6xl font-serif">
            {h.heroTitle}
          </h1>
          <p className="mt-2 text-4xl font-extrabold tracking-tight text-brand-600 dark:text-brand-400 md:text-5xl lg:text-6xl font-serif">
            {h.heroHighlight}
          </p>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-gray-600 dark:text-gray-300">
            {h.heroSub}
          </p>
          
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Link
              href="/rating-tool"
              className="group inline-flex items-center gap-2 rounded-none bg-gray-900 px-6 py-3.5 text-sm font-semibold text-white shadow-sm transition-all hover:bg-gray-800 hover:shadow-md dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100"
            >
              {h.cta}
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              href="/reports"
              className="inline-flex items-center gap-2 rounded-none px-6 py-3.5 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-800"
            >
              <Database className="h-4 w-4 text-gray-400" />
              {h.ctaSecondary}
            </Link>
          </div>
        </div>
        
        {/* Subtle decorative element for the right side of the hero */}
        <div className="hidden flex-col items-end gap-3 lg:flex">
          <div className="text-right text-sm font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">
            Rating Classes
          </div>
          <div className="flex flex-col gap-2">
            <RatingBadge rating="IG" size="md" />
            <RatingBadge rating="HY" size="md" />
            <RatingBadge rating="Distressed" size="md" />
          </div>
        </div>
      </section>

      {/* 2. Metrics Section: Asymmetric Grid */}
      <section className="rounded-none bg-gray-50/50 p-8 shadow-inner ring-1 ring-gray-200/50 dark:bg-gray-800/20 dark:ring-gray-800/50">
        <h2 className="sr-only">{h.statsTitle}</h2>
        <div className="grid grid-cols-1 gap-12 md:grid-cols-12 md:divide-x md:divide-gray-200 md:dark:divide-gray-800">
          <div className="md:col-span-6 grid grid-cols-2 gap-8">
            <div className="flex flex-col">
              <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                <Database className="h-4 w-4" />
                {h.statsRecords}
              </dt>
              <dd className="mt-4 text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
                {stats ? stats.total_records.toLocaleString() : '-'}
              </dd>
            </div>
            <div className="flex flex-col border-l border-gray-200 pl-8 dark:border-gray-800">
              <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                <BarChart2 className="h-4 w-4" />
                {h.statsCompanies}
              </dt>
              <dd className="mt-4 text-4xl font-bold tracking-tight text-brand-600 dark:text-brand-400">
                {stats ? stats.unique_tickers.toLocaleString() : '-'}
              </dd>
            </div>
          </div>
          <div className="md:col-span-3 flex flex-col px-8">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <TrendingUp className="h-4 w-4" />
              {h.statsSectors}
            </dt>
            <dd className="mt-4 text-4xl font-bold tracking-tight text-ig-dark dark:text-ig">
              {stats ? stats.unique_sectors : '-'}
            </dd>
          </div>
          <div className="md:col-span-3 flex flex-col pl-8">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Zap className="h-4 w-4" />
              {h.statsF1}
            </dt>
            <dd className="mt-4 text-4xl font-bold tracking-tight text-hy-dark dark:text-hy">
              {stats ? (stats.model_cv_f1_weighted * 100).toFixed(1) + '%' : '-'}
            </dd>
          </div>
        </div>
      </section>

      {/* 3. Value Proposition: Staggered Layout */}
      <section className="grid grid-cols-1 gap-12 md:grid-cols-12">
        <div className="md:col-span-4 group relative">
          <div className="mb-6 inline-flex h-14 w-14 items-center justify-center rounded-none bg-brand-50 text-brand-600 transition-transform duration-300 group-hover:-translate-y-1 dark:bg-brand-500/10 dark:text-brand-400">
            <Database className="h-7 w-7" />
          </div>
          <h3 className="mb-3 text-xl font-bold text-gray-900 dark:text-white font-serif">{h.step1Title}</h3>
          <p className="text-base leading-relaxed text-gray-600 dark:text-gray-400">{h.step1Desc}</p>
        </div>
        <div className="md:col-span-5 md:mt-12 group relative">
          <div className="mb-6 inline-flex h-14 w-14 items-center justify-center rounded-none bg-blue-50 text-blue-600 transition-transform duration-300 group-hover:-translate-y-1 dark:bg-blue-500/10 dark:text-blue-400">
            <BarChart2 className="h-7 w-7" />
          </div>
          <h3 className="mb-3 text-xl font-bold text-gray-900 dark:text-white font-serif">{h.step2Title}</h3>
          <p className="text-base leading-relaxed text-gray-600 dark:text-gray-400">{h.step2Desc}</p>
        </div>
        <div className="md:col-span-3 md:mt-24 group relative">
          <div className="mb-6 inline-flex h-14 w-14 items-center justify-center rounded-none bg-hy/5 text-hy transition-transform duration-300 group-hover:-translate-y-1 dark:bg-hy/10 dark:text-hy-light">
            <Zap className="h-7 w-7" />
          </div>
          <h3 className="mb-3 text-xl font-bold text-gray-900 dark:text-white font-serif">{h.step3Title}</h3>
          <p className="text-base leading-relaxed text-gray-600 dark:text-gray-400">{h.step3Desc}</p>
        </div>
      </section>

      {/* 4. Semantic Categories: Prominent Hero Category */}
      <section className="pt-8">
        <div className="mb-12 flex items-center justify-between">
          <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white font-serif">{h.groupsTitle}</h2>
          <div className="h-px flex-1 bg-gradient-to-r from-gray-200 to-transparent ml-6 dark:from-gray-800"></div>
        </div>
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
          {/* IG: The Hero Card */}
          <div className="lg:col-span-7 relative overflow-hidden rounded-none border border-ig/30 bg-ig/5 p-10 transition-all hover:bg-transparent dark:hover:bg-gray-900/50">
            <div className="mb-8 flex items-center justify-between">
              <div className="rounded-full bg-white p-3 shadow-sm dark:bg-gray-800 text-ig-dark dark:text-ig">
                <Shield className="h-8 w-8" />
              </div>
              <RatingBadge rating="IG" size="md" />
            </div>
            <h3 className="mb-4 text-3xl font-bold text-ig-dark dark:text-ig font-serif">{h.igName}</h3>
            <p className="max-w-xl text-lg leading-relaxed text-gray-700 dark:text-gray-300">{h.igDesc}</p>
          </div>
          
          {/* HY & Distressed: Sidebar Cards */}
          <div className="lg:col-span-5 flex flex-col gap-8">
            <div className="relative flex-1 overflow-hidden rounded-none border border-hy/30 bg-hy/5 p-8 transition-all hover:bg-transparent dark:hover:bg-gray-900/50">
              <div className="mb-6 flex items-center justify-between">
                <div className="rounded-full bg-white p-2.5 shadow-sm dark:bg-gray-800 text-hy-dark dark:text-hy">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <RatingBadge rating="HY" size="sm" />
              </div>
              <h3 className="mb-2 text-xl font-bold text-hy-dark dark:text-hy font-serif">{h.hyName}</h3>
              <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">{h.hyDesc}</p>
            </div>
            
            <div className="relative flex-1 overflow-hidden rounded-none border border-distressed/30 bg-distressed/5 p-8 transition-all hover:bg-transparent dark:hover:bg-gray-900/50">
              <div className="mb-6 flex items-center justify-between">
                <div className="rounded-full bg-white p-2.5 shadow-sm dark:bg-gray-800 text-distressed-dark dark:text-distressed">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <RatingBadge rating="Distressed" size="sm" />
              </div>
              <h3 className="mb-2 text-xl font-bold text-distressed-dark dark:text-distressed font-serif">{h.distName}</h3>
              <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">{h.distDesc}</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
