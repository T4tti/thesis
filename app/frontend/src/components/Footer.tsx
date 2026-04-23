'use client'

import { useLanguage } from '@/context/LanguageContext'
import { BarChart2 } from 'lucide-react'

export default function Footer() {
  const { t } = useLanguage()
  return (
    <footer className="mt-10 border-t border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      <div className="mx-auto w-full max-w-screen-2xl px-3 py-5 sm:px-4 md:px-6">
        <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
          <div className="flex items-center gap-2 text-center sm:text-left">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-500">
              <BarChart2 className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-sm font-bold text-gray-900 dark:text-white">
              VN<span className="text-brand-500">-Rating</span>
            </span>
            <span className="hidden text-sm text-gray-500 dark:text-gray-400 sm:block">- {t.footer.tagline}</span>
          </div>
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">{t.footer.copyright}</p>
        </div>
      </div>
    </footer>
  )
}
