import type { Metadata } from 'next'
import './globals.css'
import { LanguageProvider } from '@/context/LanguageContext'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'

export const metadata: Metadata = {
  title: 'VN-Rate — Corporate Credit Rating Intelligence',
  description:
    'AI-powered corporate credit rating platform. Classify companies as Investment Grade (IG), High Yield (HY), or Distressed using 12 core financial indicators.',
  keywords: 'credit rating, corporate finance, AI, LightGBM, investment grade, high yield, distressed',
  openGraph: {
    title: 'VN-Rate — Corporate Credit Intelligence',
    description: 'Instant AI-powered IG / HY / Distressed credit classification.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen flex flex-col">
        <LanguageProvider>
          <Navbar />
          <main className="flex-1 pt-16">
            {children}
          </main>
          <Footer />
        </LanguageProvider>
      </body>
    </html>
  )
}
