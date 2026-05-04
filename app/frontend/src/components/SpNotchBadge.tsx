'use client'

import React from 'react'

type SpContext = {
  indicative_notch: string
  indicative_range: string
  range_low: string
  range_high: string
  sp_scale: string[]
  migration_note: string
  disclaimer: string
  confidence_band: string
}

const NOTCH_COLOR: Record<string, string> = {
  AAA: '#10b981',
  'AA+': '#10b981',
  AA: '#10b981',
  'AA-': '#10b981',
  'A+': '#22c55e',
  A: '#22c55e',
  'A-': '#22c55e',
  'BBB+': '#84cc16',
  BBB: '#84cc16',
  'BBB-': '#eab308',
  'BB+': '#f97316',
  BB: '#f97316',
  'BB-': '#f97316',
  'B+': '#ef4444',
  B: '#ef4444',
  'B-': '#ef4444',
  'CCC+': '#dc2626',
  CCC: '#dc2626',
  'CCC-': '#dc2626',
  CC: '#991b1b',
  C: '#991b1b',
  SD: '#7f1d1d',
  D: '#7f1d1d',
}

export default function SpNotchBadge({ spContext }: { spContext: SpContext }) {
  const {
    indicative_notch,
    indicative_range,
    range_low,
    range_high,
    sp_scale,
    migration_note,
    disclaimer,
  } = spContext

  const lowIdx = sp_scale.indexOf(range_high)
  const highIdx = sp_scale.indexOf(range_low)

  const notchColor = NOTCH_COLOR[indicative_notch] ?? '#6b7280'
  const inRange = (idx: number) => {
    if (lowIdx < 0 || highIdx < 0) return false
    return idx >= lowIdx && idx <= highIdx
  }

  return (
    <div className="border-t border-gray-200 pt-6 dark:border-gray-800 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-widest text-gray-500">Indicative S&amp;P Notch</span>
        <span
          className="rounded-full px-2.5 py-1 text-xs font-mono text-white"
          style={{ backgroundColor: notchColor }}
          title={indicative_notch}
        >
          {indicative_notch}
        </span>
      </div>

      <div className="flex h-3 rounded-full overflow-hidden">
        {sp_scale.map((notch, idx) => {
          const isActive = inRange(idx)
          const isCenter = notch === indicative_notch
          const bg = NOTCH_COLOR[notch] ?? '#6b7280'
          return (
            <div
              key={notch}
              className="flex-1 origin-center"
              title={notch}
              style={{
                backgroundColor: bg,
                opacity: isActive ? 1 : 0.25,
                transform: isCenter ? 'scaleY(1.4)' : undefined,
              }}
            />
          )
        })}
      </div>

      <div className="flex items-center justify-between text-xs font-mono text-gray-400">
        <span>AAA</span>
        <span style={{ color: notchColor }}>{indicative_range}</span>
        <span>D</span>
      </div>

      <p className="italic text-xs text-gray-600 dark:text-gray-400">{migration_note}</p>
      <p className="text-[10px] text-gray-400 dark:text-gray-600">{disclaimer}</p>
    </div>
  )
}

