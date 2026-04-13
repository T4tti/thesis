type RatingType = 'IG' | 'HY' | 'Distressed'

interface Props {
  rating: string
  size?: 'sm' | 'md' | 'lg'
  showDot?: boolean
}

const CONFIG: Record<RatingType, { bg: string; text: string; border: string; dot: string; glow: string }> = {
  IG: {
    bg:     'bg-ig/10',
    text:   'text-ig-light',
    border: 'border-ig/30',
    dot:    'bg-ig',
    glow:   'shadow-glow-ig',
  },
  HY: {
    bg:     'bg-hy/10',
    text:   'text-hy-light',
    border: 'border-hy/30',
    dot:    'bg-hy',
    glow:   'shadow-glow-hy',
  },
  Distressed: {
    bg:     'bg-distressed/10',
    text:   'text-distressed-light',
    border: 'border-distressed/30',
    dot:    'bg-distressed animate-pulse2',
    glow:   'shadow-glow-dist',
  },
}

const SIZE = {
  sm: 'text-xs px-2 py-0.5 gap-1',
  md: 'text-sm px-3 py-1  gap-1.5',
  lg: 'text-base px-4 py-1.5 gap-2',
}

export default function RatingBadge({ rating, size = 'md', showDot = true }: Props) {
  const key = rating as RatingType
  const cfg = CONFIG[key] ?? CONFIG.Distressed
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
