import type { Metadata } from 'next'
import { Outfit } from 'next/font/google'
import './globals.css'
import { LanguageProvider } from '@/context/LanguageContext'
import { ThemeProvider } from '@/context/ThemeContext'
import { SidebarProvider } from '@/context/SidebarContext'
import AppShell from '@/components/AppShell'

const outfit = Outfit({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-outfit',
})

export const metadata: Metadata = {
  title: 'VN-Rating - Corporate Credit Rating Intelligence',
  description:
    'AI-powered corporate credit rating platform. Classify companies as Investment Grade (IG), High Yield (HY), or Distressed using 12 core financial indicators.',
  keywords: 'credit rating, corporate finance, AI, LightGBM, investment grade, high yield, distressed',
  openGraph: {
    title: 'VN-Rating - Corporate Credit Intelligence',
    description: 'Instant AI-powered IG / HY / Distressed credit classification.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={outfit.variable}>
      <body className="min-h-screen bg-gray-50 text-gray-900 antialiased dark:bg-gray-950 dark:text-gray-100 font-sans">
        <LanguageProvider>
          <ThemeProvider>
            <SidebarProvider>
              <AppShell>{children}</AppShell>
            </SidebarProvider>
          </ThemeProvider>
        </LanguageProvider>
      </body>
    </html>
  )
}
