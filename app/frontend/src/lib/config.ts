/**
 * Shared runtime configuration constants.
 * All API calls must reference API_BASE_URL from this module —
 * never hardcode the fallback URL inline.
 */
const BROWSER_API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const SERVER_API_BASE_URL =
  process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://backend:8000'

export const API_BASE_URL =
  typeof window === 'undefined' ? SERVER_API_BASE_URL : BROWSER_API_BASE_URL
