'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { en } from '@/locales/en'
import { vi } from '@/locales/vi'
import type { Translations } from '@/locales/en'

type Lang = 'en' | 'vi'

interface LanguageContextValue {
  lang: Lang
  setLang: (l: Lang) => void
  t: Translations
  toggle: () => void
}

const LanguageContext = createContext<LanguageContextValue>({
  lang: 'en',
  setLang: () => {},
  t: en,
  toggle: () => {},
})

const LOCALES: Record<Lang, Translations> = { en, vi }
const STORAGE_KEY = 'vnrate-lang'

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>('en')

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as Lang | null
    if (saved === 'vi' || saved === 'en') setLangState(saved)
  }, [])

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const setLang = (l: Lang) => {
    setLangState(l)
    localStorage.setItem(STORAGE_KEY, l)
  }

  const toggle = () => setLang(lang === 'en' ? 'vi' : 'en')

  return (
    <LanguageContext.Provider value={{ lang, setLang, t: LOCALES[lang], toggle }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  return useContext(LanguageContext)
}
