'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import StatsCard from '@/components/StatsCard'
import RatingBadge from '@/components/RatingBadge'
import { ArrowRight, BarChart2, Database, TrendingUp, Zap, Shield, AlertTriangle } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Stats {
  total_records: number
  unique_tickers: number
  unique_sectors: number
  model_cv_f1_weighted: number
}

export default function HomePage() {
  const { t } = useLanguage()
  const h = t.home
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    fetch(`${API}/api/stats`)
      .then(r => r.json())
      .then(d => setStats(d))
      .catch(() => {})
  }, [])

  return (
    <>
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <section className="relative min-h-[92vh] flex items-center overflow-hidden">
        {/* Background grid */}
        <div className="absolute inset-0 hero-grid opacity-60" />

        {/* Blobs */}
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-primary-600/20 rounded-full blur-3xl blob" />
        <div className="absolute top-40 right-1/4 w-72 h-72 bg-sky-500/15 rounded-full blur-3xl blob blob-delay-2" />
        <div className="absolute bottom-20 left-1/3 w-64 h-64 bg-primary-800/20 rounded-full blur-3xl blob blob-delay-4" />

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary-500/30 bg-primary-500/10 text-primary-300 text-sm font-medium mb-8 animate-fade-in-up">
            <Zap className="w-3.5 h-3.5" />
            {h.badge}
          </div>

          {/* Heading */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6 animate-fade-in-up" style={{animationDelay:'0.1s'}}>
            {h.heroTitle}
            <br />
            <span className="gradient-text">{h.heroHighlight}</span>
          </h1>

          {/* Subtitle */}
          <p className="text-slate-400 text-lg sm:text-xl max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in-up" style={{animationDelay:'0.2s'}}>
            {h.heroSub}
          </p>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in-up" style={{animationDelay:'0.3s'}}>
            <Link
              href="/rating-tool"
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl bg-primary-600 hover:bg-primary-500 text-white font-semibold text-sm transition-all shadow-glow-primary"
            >
              {h.cta}
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/reports"
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl border border-white/15 hover:border-primary-500/50 hover:bg-primary-500/5 text-slate-300 hover:text-white font-semibold text-sm transition-all"
            >
              <Database className="w-4 h-4" />
              {h.ctaSecondary}
            </Link>
          </div>

          {/* Floating rating badges preview */}
          <div className="flex justify-center gap-3 mt-14 animate-fade-in-up" style={{animationDelay:'0.4s'}}>
            {(['IG', 'HY', 'Distressed'] as const).map((r, i) => (
              <div key={r} className="animate-float" style={{animationDelay:`${i * 0.5}s`}}>
                <RatingBadge rating={r} size="md" />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Stats ─────────────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-24">
        <h2 className="text-center text-sm font-semibold uppercase tracking-widest text-slate-500 mb-10">
          {h.statsTitle}
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatsCard
            icon={<Database className="w-5 h-5" />}
            label={h.statsRecords}
            value={stats ? stats.total_records.toLocaleString() : '—'}
          />
          <StatsCard
            icon={<BarChart2 className="w-5 h-5" />}
            label={h.statsCompanies}
            value={stats ? stats.unique_tickers.toLocaleString() : '—'}
            accent="text-sky-400"
          />
          <StatsCard
            icon={<TrendingUp className="w-5 h-5" />}
            label={h.statsSectors}
            value={stats ? stats.unique_sectors : '—'}
            accent="text-ig-light"
          />
          <StatsCard
            icon={<Zap className="w-5 h-5" />}
            label={h.statsF1}
            value={stats ? (stats.model_cv_f1_weighted * 100).toFixed(1) + '%' : '—'}
            accent="text-hy-light"
          />
        </div>
      </section>

      {/* ── How it works ──────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-28">
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">{h.howTitle}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { num: '01', title: h.step1Title, desc: h.step1Desc, icon: <Database className="w-6 h-6" /> },
            { num: '02', title: h.step2Title, desc: h.step2Desc, icon: <BarChart2 className="w-6 h-6" /> },
            { num: '03', title: h.step3Title, desc: h.step3Desc, icon: <Zap className="w-6 h-6" /> },
          ].map((step, i) => (
            <div key={i} className="glass-card rounded-2xl p-7 group hover:border-primary-500/25 transition-all duration-300">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-primary-500/15 flex items-center justify-center text-primary-400 shrink-0 group-hover:bg-primary-500/25 transition-colors">
                  {step.icon}
                </div>
                <div>
                  <div className="text-xs font-mono text-primary-500 mb-1">{step.num}</div>
                  <h3 className="font-semibold text-white mb-2">{step.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{step.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Rating groups ─────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-28">
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">{h.groupsTitle}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* IG */}
          <div className="rounded-2xl border border-ig/25 bg-ig-gradient p-8 group hover:border-ig/50 hover:shadow-glow-ig transition-all duration-300">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-ig/15 flex items-center justify-center">
                <Shield className="w-5 h-5 text-ig" />
              </div>
              <RatingBadge rating="IG" size="md" />
            </div>
            <h3 className="text-xl font-bold text-ig-light mb-2">{h.igName}</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{h.igDesc}</p>
          </div>

          {/* HY */}
          <div className="rounded-2xl border border-hy/25 bg-hy-gradient p-8 group hover:border-hy/50 hover:shadow-glow-hy transition-all duration-300">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-hy/15 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-hy" />
              </div>
              <RatingBadge rating="HY" size="md" />
            </div>
            <h3 className="text-xl font-bold text-hy-light mb-2">{h.hyName}</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{h.hyDesc}</p>
          </div>

          {/* Distressed */}
          <div className="rounded-2xl border border-distressed/25 bg-dist-gradient p-8 group hover:border-distressed/50 hover:shadow-glow-dist transition-all duration-300">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-distressed/15 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-distressed" />
              </div>
              <RatingBadge rating="Distressed" size="md" />
            </div>
            <h3 className="text-xl font-bold text-distressed-light mb-2">{h.distName}</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{h.distDesc}</p>
          </div>
        </div>

        {/* Final CTA */}
        <div className="mt-16 text-center">
          <Link
            href="/rating-tool"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-primary-600 hover:bg-primary-500 text-white font-semibold transition-all shadow-glow-primary animate-glow text-base"
          >
            {h.cta}
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </>
  )
}
