/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#eef2ff',
          100: '#e0e7ff',
          200: '#c7d2fe',
          300: '#a5b4fc',
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
          800: '#3730a3',
          900: '#312e81',
        },
        sky: {
          400: '#38bdf8',
          500: '#0ea5e9',
        },
        surface: {
          100: '#1e2035',
          200: '#161828',
          300: '#0f1020',
          400: '#090a18',
        },
        ig:          { DEFAULT: '#10b981', light: '#34d399', dark: '#059669', glow: 'rgba(16,185,129,0.25)' },
        hy:          { DEFAULT: '#f59e0b', light: '#fbbf24', dark: '#d97706', glow: 'rgba(245,158,11,0.25)' },
        distressed:  { DEFAULT: '#ef4444', light: '#f87171', dark: '#dc2626', glow: 'rgba(239,68,68,0.25)'  },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        'fade-in-up': {
          '0%':   { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)'    },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)'   },
          '50%':       { transform: 'translateY(-8px)'  },
        },
        pulse2: {
          '0%, 100%': { opacity: '1',   transform: 'scale(1)'    },
          '50%':       { opacity: '0.7', transform: 'scale(0.97)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition:  '1000px 0' },
        },
        'spin-slow': {
          to: { transform: 'rotate(360deg)' },
        },
        glow: {
          '0%,100%': { boxShadow: '0 0 16px rgba(99,102,241,0.3)' },
          '50%':      { boxShadow: '0 0 36px rgba(99,102,241,0.7)' },
        },
      },
      animation: {
        'fade-in-up': 'fade-in-up 0.5s ease-out both',
        float:        'float 4s ease-in-out infinite',
        pulse2:       'pulse2 2s ease-in-out infinite',
        shimmer:      'shimmer 2s linear infinite',
        'spin-slow':  'spin-slow 8s linear infinite',
        glow:         'glow 2.5s ease-in-out infinite',
      },
      backgroundImage: {
        'hero-gradient':   'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.25), transparent)',
        'card-gradient':   'linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%)',
        'ig-gradient':     'linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05))',
        'hy-gradient':     'linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05))',
        'dist-gradient':   'linear-gradient(135deg, rgba(239,68,68,0.2) , rgba(239,68,68,0.05))',
      },
      boxShadow: {
        'card': '0 1px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
        'glow-primary': '0 0 30px rgba(99,102,241,0.4)',
        'glow-ig':      '0 0 20px rgba(16,185,129,0.4)',
        'glow-hy':      '0 0 20px rgba(245,158,11,0.4)',
        'glow-dist':    '0 0 20px rgba(239,68,68,0.4)',
      },
    },
  },
  plugins: [],
}
