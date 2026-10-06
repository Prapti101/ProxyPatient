import { useEffect, useState } from 'react'
import { ArrowUpRight, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import type { ValidationResponse } from '../types/api'
import { ErrorBox, Loading, PageHeading, StatusBadge } from '../components/ui/Blocks'

function MetricTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value == null) return <span className="null-value">Not reported</span>
  if (typeof value === 'number')
    return (
      <span>
        {Number.isFinite(value)
          ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
          : 'Not reported'}
      </span>
    )
  if (typeof value === 'string' || typeof value === 'boolean') return <span>{String(value)}</span>
  if (Array.isArray(value))
    return (
      <ul className="metric-list">
        {value.map((item, index) => (
          <li key={index}>
            <MetricTree value={item} depth={depth + 1} />
          </li>
        ))}
      </ul>
    )
  if (typeof value === 'object')
    return (
      <dl className="metric-tree">
        {Object.entries(value as Record<string, unknown>).map(([key, item]) => (
          <div
            className={`metric-entry ${item !== null && typeof item === 'object' ? 'nested' : ''}`}
            key={key}
          >
            <dt>{key.replace(/_/g, ' ')}</dt>
            <dd>
              {item !== null && typeof item === 'object' && depth > 0 ? (
                <details>
                  <summary>View details</summary>
                  <MetricTree value={item} depth={depth + 1} />
                </details>
              ) : (
                <MetricTree value={item} depth={depth + 1} />
              )}
            </dd>
          </div>
        ))}
      </dl>
    )
  return null
}
function ReportPanel({
  title,
  data,
  error,
  retry,
  comparison = false,
}: {
  title: string
  data: ValidationResponse | null
  error: string
  retry: () => void
  comparison?: boolean
}) {
  if (error)
    return (
      <section className="report-panel">
        <h2>{title}</h2>
        <ErrorBox message={error} onRetry={retry} />
      </section>
    )
  if (!data) return null
  const contents = comparison ? (data.models ?? data.metrics) : data.metrics
  return (
    <section className="report-panel">
      <div className="report-panel-heading">
        <div>
          <span className="eyebrow">{comparison ? '02 / Across models' : '01 / Evaluation'}</span>
          <h2>{title}</h2>
        </div>
        <span className={`report-state ${data.status === 'complete' ? 'complete' : ''}`}>
          {data.status === 'complete' ? 'Report available' : 'Pending'}
        </span>
      </div>
      <div className="report-provenance">
        <StatusBadge
          status={data.status_banner}
          runType={data.run_type}
          preliminary={data.preliminary}
          demo={data.demo}
        />
        <p>{data.note}</p>
      </div>
      {data.status !== 'complete' ? (
        <div className="report-pending">
          <span aria-hidden="true">—</span>
          <div>
            <h3>{comparison ? 'No measured comparison yet.' : 'The evidence is still pending.'}</h3>
            <p>
              A compatible measured report has not been supplied. There are no substitute scores to
              display.
            </p>
          </div>
        </div>
      ) : contents ? (
        <div className="report-card">
          <MetricTree value={contents} />
        </div>
      ) : (
        <p className="notice">The report is marked complete but contains no metrics.</p>
      )}
      {data.evaluation_scope && (
        <details className="report-scope">
          <summary>Evaluation scope</summary>
          <MetricTree value={data.evaluation_scope} />
        </details>
      )}
      {data.model_fingerprint && (
        <details className="report-scope">
          <summary>Model identity</summary>
          <p className="fingerprint">{data.model_fingerprint}</p>
        </details>
      )}
    </section>
  )
}
export default function Validation() {
  const [reports, setReports] = useState<(ValidationResponse | null)[]>([null, null])
  const [errors, setErrors] = useState(['', ''])
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let active = true
    setLoading(true)
    Promise.allSettled([api.getValidation(), api.getModelComparison()]).then((results) => {
      if (!active) return
      setReports(results.map((result) => (result.status === 'fulfilled' ? result.value : null)))
      setErrors(
        results.map((result) =>
          result.status === 'rejected'
            ? result.reason instanceof Error
              ? result.reason.message
              : 'Could not load this report.'
            : '',
        ),
      )
      setLoading(false)
    })
    return () => {
      active = false
    }
  }, [attempt])
  const retry = () => setAttempt((value) => value + 1)
  return (
    <div className="page">
      <PageHeading number="04" title="Read the evidence.">
        Measured reports, their scope, and the limits of what they can tell us.
      </PageHeading>
      <div className="evidence-intro">
        <p>
          Validation describes model behavior. A complete report is evidence to inspect, not a
          guarantee of accuracy or privacy.
        </p>
        <button className="button secondary" onClick={retry} disabled={loading}>
          <RefreshCw size={15} />
          Refresh reports
        </button>
      </div>
      {loading ? (
        <Loading label="Loading measured reports…" />
      ) : (
        <>
          <ReportPanel title="Model validation" data={reports[0]} error={errors[0]} retry={retry} />
          <ReportPanel
            title="Model comparison"
            data={reports[1]}
            error={errors[1]}
            retry={retry}
            comparison
          />
        </>
      )}
      <aside className="reading-note">
        <span className="eyebrow">Before drawing a conclusion</span>
        <h3>Read the scope with the score.</h3>
        <p>
          Look at retained samples, run type and subgroup support. Missing or suppressed values stay
          unreported. No model is declared a winner from its architecture alone.
        </p>
        <Link to="/how-it-works" className="text-link">
          How evaluation fits the method <ArrowUpRight size={15} />
        </Link>
      </aside>
    </div>
  )
}
