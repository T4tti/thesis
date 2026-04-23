type RatingType = 'IG' | 'HY' | 'Distressed' | 'unknown'

interface Props {
  rating: string
  size?: 'sm' | 'md' | 'lg'
  showDot?: boolean
}

const CONFIG: Record<RatingType, { bg: string; text: string; border: string; dot: string }> = {
  IG: {
    bg: 'bg-ig/10',
    text: 'text-ig-dark dark:text-ig',
    border: 'border-ig/30',
    dot: 'bg-ig',
  },
  HY: {
    bg: 'bg-hy/10',
    text: 'text-hy-dark dark:text-hy',
    border: 'border-hy/30',
    dot: 'bg-hy',
  },
  Distressed: {
    bg: 'bg-distressed/10',
    text: 'text-distressed-dark dark:text-distressed',
    border: 'border-distressed/30',
    dot: 'bg-distressed',
  },
  unknown: {
    bg: 'bg-gray-100 dark:bg-gray-800',
    text: 'text-gray-600 dark:text-gray-300',
    border: 'border-gray-300 dark:border-gray-700',
    dot: 'bg-gray-400',
  },
}

/**
 * Maps granular letter-grade ratings to the 3-group system.
 * Exact group names (IG / HY / Distressed) pass through unchanged.
 */
const resolveRatingGroup = (rating: string): RatingType => {
  const r = rating.trim().toUpperCase()
  if (r === 'IG') return 'IG'
  if (r === 'HY') return 'HY'
  if (r === 'DISTRESSED') return 'Distressed'
  if (/^(AAA|AA[+-]?|A[+-]?|BBB[+-]?)$/.test(r)) return 'IG'
  if (/^(BB[+-]?|B[+-]?)$/.test(r)) return 'HY'
  if (/^(CCC[+-]?|CC|C|D)$/.test(r)) return 'Distressed'
  return 'unknown'
}

const SIZE = {
  sm: 'text-xs px-2 py-0.5 gap-1',
  md: 'text-sm px-3 py-1  gap-1.5',
  lg: 'text-base px-4 py-1.5 gap-2',
}

export default function RatingBadge({ rating, size = 'md', showDot = true }: Props) {
  const key = resolveRatingGroup(rating)
  const cfg = CONFIG[key]
  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full border
        ${cfg.bg} ${cfg.text} ${cfg.border} ${SIZE[size]}`}
    >
      {showDot && (
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
      )}
      {rating}
    </span>
  )
}
