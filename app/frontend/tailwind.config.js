/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          25: '#f2f7ff',
          50: '#ecf3ff',
          100: '#dde9ff',
          200: '#c2d6ff',
          300: '#9cb9ff',
          400: '#7592ff',
          500: '#465fff',
          600: '#3641f5',
          700: '#2a31d8',
          800: '#252dae',
          900: '#262e89',
          950: '#161950',
        },
        ig: { DEFAULT: '#12b76a', light: '#32d583', dark: '#027a48' },
        hy: { DEFAULT: '#f79009', light: '#fdb022', dark: '#b54708' },
        distressed: { DEFAULT: '#f04438', light: '#f97066', dark: '#b42318' },
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'Be Vietnam Pro', 'system-ui', 'sans-serif'],
        serif: ['var(--font-serif)', 'Lora', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
