import { AlertTriangle, ArrowRight, LoaderCircle, RefreshCw, Info } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

export const Disclaimer = () => (
  <aside className="disclaimer">
    <Info size={17} />
    <p>
      For research and education. Synthetic group summaries are not a diagnosis, treatment advice or
      a personal prediction. What-if comparisons are descriptive, not causal.
    </p>
  </aside>
)
export function StatusBadge({
  status,
  runType,
  preliminary,
  demo,
}: {
  status?: string
  runType?: string | null
  preliminary?: boolean
  demo?: boolean
}) {
  const label =
    status ??
    (demo || runType === 'mock'
      ? 'DEMO (mock data)'
      : preliminary || runType === 'quick'
        ? 'PRELIMINARY (quick run)'
        : runType === 'full'
          ? 'FULL RUN'
          : 'STATUS UNAVAILABLE')
  return (
    <span
      className={`status-badge ${demo || runType === 'mock' ? 'demo' : preliminary || runType === 'quick' ? 'preliminary' : runType === 'full' ? 'full' : 'unknown'}`}
    >
      <i aria-hidden="true" />
      {label}
    </span>
  )
}
export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="state-box" role="status">
      <LoaderCircle className="spin" size={19} />
      <span>{label}</span>
    </div>
  )
}
export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-box" role="alert">
      <AlertTriangle size={20} />
      <div>
        <b>We couldn’t complete that request</b>
        <p>{message}</p>
        {onRetry && (
          <button className="text-button" onClick={onRetry}>
            <RefreshCw size={14} /> Try again
          </button>
        )}
      </div>
    </div>
  )
}
export function Empty({
  title,
  children,
  action,
}: {
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty-box">
      <span className="empty-mark" aria-hidden="true">
        —
      </span>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}
export function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string
  title: string
  children?: ReactNode
}) {
  return (
    <div className="section-title">
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  )
}
export function PageHeading({
  number,
  title,
  children,
}: {
  number: string
  title: string
  children: ReactNode
}) {
  return (
    <header className="page-heading">
      <div>
        <span className="eyebrow">Research workspace / {number}</span>
        <h1>{title}</h1>
        <p>{children}</p>
      </div>
      <span className="page-index" aria-hidden="true">
        {number}
      </span>
    </header>
  )
}
export function ExploreLink() {
  return (
    <Link to="/explore" className="button primary">
      Build a reference profile <ArrowRight size={16} />
    </Link>
  )
}
