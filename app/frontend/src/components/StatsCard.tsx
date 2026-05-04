import { ReactNode } from 'react'
import { Card } from '@/components/common/Card'

interface Props {
  icon: ReactNode
  label: string
  value: string | number
  sub?: string
  accent?: string
}

export default function StatsCard({ icon, label, value, sub, accent = 'text-brand-500' }: Props) {
  return (
    <Card padding="sm" hover>
      <div>
        <div className={`mb-3 flex h-10 w-10 items-center justify-center rounded-none bg-brand-50 dark:bg-brand-500/15 ${accent}`}>
          {icon}
        </div>
        <div className={`mb-1 text-2xl font-bold tracking-tight ${accent}`}>{value}</div>
        <div className="text-sm font-medium text-gray-700 dark:text-gray-200">{label}</div>
        {sub && <div className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{sub}</div>}
      </div>
    </Card>
  )
}
