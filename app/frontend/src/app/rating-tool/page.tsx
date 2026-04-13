'use client'

import { useLanguage } from '@/context/LanguageContext'
import RatingForm from '@/components/RatingForm'
import { Cpu, Info } from 'lucide-react'

export default function RatingToolPage() {
  const { t } = useLanguage()
  const tr = t.ratingTool

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-2 text-primary-400 text-sm font-medium mb-3">
          <Cpu className="w-4 h-4" />
          <span className="uppercase tracking-widest text-xs">{tr.title}</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-3">{tr.title}</h1>
        <p className="text-slate-400 max-w-xl">{tr.subtitle}</p>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-3 bg-primary-500/8 border border-primary-500/20 rounded-xl p-4 mb-10 max-w-3xl">
        <Info className="w-4 h-4 text-primary-400 shrink-0 mt-0.5" />
        <p className="text-slate-300 text-sm leading-relaxed">{tr.missingNote}</p>
      </div>

      <RatingForm />
    </div>
  )
}
