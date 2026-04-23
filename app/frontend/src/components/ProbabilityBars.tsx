const COLOR_CLASSES: Record<string, string> = {
  IG:         'bg-ig',
  HY:         'bg-hy',
  Distressed: 'bg-distressed',
}

const ORDER = ['IG', 'HY', 'Distressed']

interface Props {
  probabilities: Record<string, number>
}

const toClampedPercent = (value: number) => {
  const finite = Number.isFinite(value) ? value : 0
  return Math.max(0, Math.min(100, finite * 100))
}

export default function ProbabilityBars({ probabilities }: Props) {
  const sorted = ORDER.filter(k => probabilities[k] !== undefined).concat(
    Object.keys(probabilities).filter(k => !ORDER.includes(k))
  )

  return (
    <div className="space-y-3">
      {sorted.map(cls => {
        const pct = toClampedPercent(probabilities[cls] ?? 0)
        const bgClass = COLOR_CLASSES[cls] ?? 'bg-gray-400' // neutral gray for unknown ratings
        return (
          <div key={cls}>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-gray-700 dark:text-gray-200">{cls}</span>
              <span className="font-mono text-gray-500 dark:text-gray-400">{pct.toFixed(1)}%</span>
            </div>
            <div className="relative h-2.5 overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
              <div
                className={`absolute inset-0 h-full w-full rounded-full transition-transform duration-700 ease-out ${bgClass}`}
                style={{
                  transform: `translateX(-${100 - pct}%)`
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
