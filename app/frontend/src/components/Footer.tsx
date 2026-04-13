import { useLanguage } from '@/context/LanguageContext'
import { BarChart2 } from 'lucide-react'

export default function Footer() {
  const { t } = useLanguage()
  return (
    <footer className="border-t border-white/5 bg-surface-400/60 mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center">
              <BarChart2 className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold text-white">VN<span className="text-primary-400">-Rate</span></span>
            <span className="text-slate-500 text-sm hidden sm:block">— {t.footer.tagline}</span>
          </div>
          <p className="text-slate-500 text-sm text-center">{t.footer.copyright}</p>
        </div>
      </div>
    </footer>
  )
}
