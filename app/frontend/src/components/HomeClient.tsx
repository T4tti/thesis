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
          <h1 className="text-4xl font-light tracking-tight text-gray-900 dark:text-white md:text-5xl lg:text-6xl">
            {h.heroTitle}
          </h1>
          <p className="mt-2 text-4xl font-extrabold tracking-tight text-brand-600 dark:text-brand-400 md:text-5xl lg:text-6xl">
            {h.heroHighlight}
          </p>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-gray-600 dark:text-gray-300">
            {h.heroSub}
          </p>
          
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Link
              href="/rating-tool"
              className="group inline-flex items-center gap-2 rounded-xl bg-gray-900 px-6 py-3.5 text-sm font-semibold text-white shadow-sm transition-all hover:bg-gray-800 hover:shadow-md dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100"
            >
              {h.cta}
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              href="/reports"
              className="inline-flex items-center gap-2 rounded-xl px-6 py-3.5 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-800"
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

      {/* 2. Metrics Section: Unified Analytical Strip instead of generic cards */}
      <section className="rounded-2xl bg-gray-50/50 p-8 shadow-inner ring-1 ring-gray-200/50 dark:bg-gray-800/20 dark:ring-gray-800/50">
        <h2 className="sr-only">{h.statsTitle}</h2>
        <div className="grid grid-cols-2 gap-y-8 divide-x divide-gray-200 dark:divide-gray-800 md:grid-cols-4">
          <div className="flex flex-col px-6 first:pl-0">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Database className="h-4 w-4" />
              {h.statsRecords}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              {stats ? stats.total_records.toLocaleString() : '-'}
            </dd>
          </div>
          <div className="flex flex-col px-6">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <BarChart2 className="h-4 w-4" />
              {h.statsCompanies}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-blue-600 dark:text-blue-400">
              {stats ? stats.unique_tickers.toLocaleString() : '-'}
            </dd>
          </div>
          <div className="flex flex-col px-6">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <TrendingUp className="h-4 w-4" />
              {h.statsSectors}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-ig-dark dark:text-ig">
              {stats ? stats.unique_sectors : '-'}
            </dd>
          </div>
          <div className="flex flex-col px-6 last:pr-0">
            <dt className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400">
              <Zap className="h-4 w-4" />
              {h.statsF1}
            </dt>
            <dd className="mt-2 text-3xl font-bold tracking-tight text-hy-dark dark:text-hy">
              {stats ? (stats.model_cv_f1_weighted * 100).toFixed(1) + '%' : '-'}
            </dd>
          </div>
        </div>
      </section>

      {/* 3. Value Proposition / How it works */}
      <section className="grid grid-cols-1 gap-8 md:grid-cols-3">
        {[
          { title: h.step1Title, desc: h.step1Desc, icon: <Database className="h-6 w-6" /> },
          { title: h.step2Title, desc: h.step2Desc, icon: <BarChart2 className="h-6 w-6" /> },
          { title: h.step3Title, desc: h.step3Desc, icon: <Zap className="h-6 w-6" /> },
        ].map((item, i) => (
          <div key={item.title} className="group relative">
            <div className="mb-5 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50 text-brand-600 transition-transform duration-300 group-hover:-translate-y-1 dark:bg-brand-500/10 dark:text-brand-400">
              {item.icon}
            </div>
            <h3 className="mb-2 text-lg font-bold text-gray-900 dark:text-white">{item.title}</h3>
            <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">{item.desc}</p>
          </div>
        ))}
      </section>

      {/* 4. Semantic Categories */}
      <section className="pt-8">
        <div className="mb-8 flex items-center justify-between">
          <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white">{h.groupsTitle}</h2>
          <div className="h-px flex-1 bg-gradient-to-r from-gray-200 to-transparent ml-6 dark:from-gray-800"></div>
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {[
            { name: h.igName, desc: h.igDesc, rating: 'IG', border: 'border-ig/30', bg: 'bg-ig/5', text: 'text-ig-dark dark:text-ig', icon: <Shield className="h-5 w-5" /> },
            { name: h.hyName, desc: h.hyDesc, rating: 'HY', border: 'border-hy/30', bg: 'bg-hy/5', text: 'text-hy-dark dark:text-hy', icon: <TrendingUp className="h-5 w-5" /> },
            { name: h.distName, desc: h.distDesc, rating: 'Distressed', border: 'border-distressed/30', bg: 'bg-distressed/5', text: 'text-distressed-dark dark:text-distressed', icon: <AlertTriangle className="h-5 w-5" /> },
          ].map((cat) => (
            <div key={cat.name} className={`relative overflow-hidden rounded-2xl border ${cat.border} ${cat.bg} p-8 transition-colors hover:bg-transparent dark:hover:bg-gray-900/50`}>
              <div className="mb-6 flex items-center justify-between">
                <div className={`rounded-full bg-white p-2.5 shadow-sm dark:bg-gray-800 ${cat.text}`}>
                  {cat.icon}
                </div>
                <RatingBadge rating={cat.rating as any} size="sm" />
              </div>
              <h3 className={`mb-3 text-xl font-bold ${cat.text}`}>{cat.name}</h3>
              <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">{cat.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
