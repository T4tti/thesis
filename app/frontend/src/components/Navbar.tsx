'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { useLanguage } from '@/context/LanguageContext'
import { BarChart2, Menu, X, Globe } from 'lucide-react'

export default function Navbar() {
  const { t, toggle, lang } = useLanguage()
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

  const links = [
    { href: '/',             label: t.nav.home        },
    { href: '/methodology',  label: t.nav.methodology },
    { href: '/reports',      label: t.nav.reports     },
    { href: '/rating-tool',  label: t.nav.ratingTool  },
  ]

  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href)

  return (
    <nav className="fixed top-0 inset-x-0 z-50 border-b border-white/5 bg-surface-400/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">

          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-glow-primary group-hover:shadow-glow-primary transition-shadow">
              <BarChart2 className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-xl tracking-tight text-white">
              VN<span className="text-primary-400">-Rate</span>
            </span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-1">
            {links.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive(href)
                    ? 'bg-primary-500/20 text-primary-300'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            {/* Language toggle */}
            <button
              onClick={toggle}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-white/5 transition-all border border-white/10"
            >
              <Globe className="w-3.5 h-3.5" />
              <span className="hidden sm:block">{lang === 'en' ? 'VI' : 'EN'}</span>
            </button>

            {/* CTA */}
            <Link
              href="/rating-tool"
              className="hidden md:flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary-600 hover:bg-primary-500 text-white text-sm font-semibold transition-colors shadow-glow-primary"
            >
              {t.nav.ratingTool}
            </Link>

            {/* Mobile menu */}
            <button
              className="md:hidden p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
              onClick={() => setOpen(!open)}
              aria-label="Toggle menu"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden border-t border-white/5 bg-surface-300/95 backdrop-blur-xl">
          <div className="px-4 py-3 space-y-1">
            {links.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={`block px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive(href)
                    ? 'bg-primary-500/20 text-primary-300'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  )
}
