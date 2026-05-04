import React from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
  hover?: boolean
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

export const Card: React.FC<CardProps> = ({ children, hover, padding = 'md', className = '', ...props }) => {
  const baseClasses = 'rounded-none border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900'
  const hoverClasses = hover ? 'transition-all hover:border-brand-200 hover:shadow-md dark:hover:border-brand-500/30' : ''
  
  const paddingClasses = {
    none: '',
    sm: 'p-4 sm:p-5',
    md: 'p-5 sm:p-6',
    lg: 'p-6 sm:p-8',
  }[padding]

  return (
    <section className={`${baseClasses} ${hoverClasses} ${paddingClasses} ${className}`.trim()} {...props}>
      {children}
    </section>
  )
}
