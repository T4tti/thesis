'use client'

import { useLanguage } from '@/context/LanguageContext'
import RatingForm from '@/components/RatingForm'
import { BarChart3 } from 'lucide-react'

export default function RatingToolPage() {
  const { t } = useLanguage()
  const tr = t.ratingTool

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="flex flex-col gap-6 border-b border-gray-200 pb-10 dark:border-gray-800">
        <div className="max-w-3xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-600 dark:border-brand-500/30 dark:bg-brand-500/10 dark:text-brand-300">
            <BarChart3 className="h-3.5 w-3.5" />
            {tr.title}
          </div>
          <h1 className="text-4xl font-light tracking-tight text-gray-900 dark:text-white md:text-5xl lg:text-6xl">
            {tr.title}
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-gray-600 dark:text-gray-300">
            {tr.subtitle}
          </p>
        </div>
      </header>

      <RatingForm />
    </div>
  )
}
