const base = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
export class HttpError extends Error { constructor(message: string, public status?: number) { super(message); this.name = 'HttpError' } }
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`${base}${path}`, { ...init, headers: { ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers } }) }
  catch { throw new HttpError('Could not reach the ProxyPatient API. Check that it is running and the API URL is correct.') }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    const message = Array.isArray(detail) ? detail.map((e: { loc?: (string|number)[]; msg?: string }) => `${e.loc?.slice(-1)[0] ?? 'Request'}: ${e.msg ?? 'Invalid value'}`).join(' · ') : typeof detail === 'string' ? detail : `Unexpected API response (${response.status}).`
    throw new HttpError(response.status === 503 ? `The model is currently unavailable. ${message}` : message, response.status)
  }
  if (body === null || typeof body !== 'object') throw new HttpError('The API returned an unexpected response. Please try again.')
  return body as T
}
