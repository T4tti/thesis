/**
 * Typed API fetch wrapper.
 *
 * - Prefixes `NEXT_PUBLIC_API_URL`
 * - Injects `Accept-Language` header on every request
 * - Throws on non-OK responses with FastAPI `detail` when available
 */

export type Lang = 'en' | 'vi'

type FastApiError = {
  detail?: unknown
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const joinUrl = (base: string, path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${base.replace(/\/+$/, '')}${normalizedPath}`
}

export const apiFetch = async <T>(
  path: string,
  options: RequestInit = {},
  lang: Lang = 'en',
): Promise<T> => {
  const headers = new Headers(options.headers)
  headers.set('Accept-Language', lang)

  const res = await fetch(joinUrl(API_URL, path), {
    ...options,
    headers,
  })

  const contentType = res.headers.get('content-type') || ''

  if (!res.ok) {
    let detail: string | undefined

    if (contentType.includes('application/json')) {
      try {
        const data: unknown = await res.json()
        const maybe = data as FastApiError
        if (maybe && typeof maybe === 'object' && 'detail' in maybe && maybe.detail != null) {
          detail = String(maybe.detail)
        }
      } catch {
        // ignore JSON parse errors
      }
    } else {
      try {
        const text = await res.text()
        if (text) detail = text
      } catch {
        // ignore read errors
      }
    }

    throw new Error(detail || res.statusText || `Request failed (${res.status})`)
  }

  if (contentType.includes('application/json')) {
    return (await res.json()) as T
  }

  return (await res.text()) as unknown as T
}

