const COLORS: Record<string, string> = {
  IG:         '#10b981',
  HY:         '#f59e0b',
  Distressed: '#ef4444',
}

const ORDER = ['IG', 'HY', 'Distressed']

interface Props {
  probabilities: Record<string, number>
}

export default function ProbabilityBars({ probabilities }: Props) {
  const sorted = ORDER.filter(k => probabilities[k] !== undefined).concat(
    Object.keys(probabilities).filter(k => !ORDER.includes(k))
  )

  return (
    <div className="space-y-3">
      {sorted.map(cls => {
        const pct = (probabilities[cls] ?? 0) * 100
        const color = COLORS[cls] ?? '#6366f1'
        return (
          <div key={cls}>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-300">{cls}</span>
              <span className="text-slate-400 font-mono">{pct.toFixed(1)}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-surface-100/40 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${pct}%`,
                  backgroundColor: color,
                  boxShadow: `0 0 6px ${color}80`,
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
