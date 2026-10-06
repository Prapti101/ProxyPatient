import { useRef, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import { useAppState } from '../context/AppState'
import {
  ConditionDetails,
  Examples,
  MetricCard,
  ProfileForm,
} from '../components/scenarios/ScenarioUI'
import { Disclaimer, ErrorBox, PageHeading } from '../components/ui/Blocks'
import type { Condition } from '../types/api'

export default function Explore() {
  const state = useAppState()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [submitted, setSubmitted] = useState<Condition | null>(null)
  const resultHeading = useRef<HTMLHeadingElement>(null)
  const result = state.baselineResult
  async function generate(condition: Condition) {
    setSubmitted(condition)
    setError('')
    setBusy(true)
    try {
      const response = await api.generate({ condition, n: 1000, n_examples: 3 })
      state.acceptBaseline(condition, response)
      requestAnimationFrame(() => resultHeading.current?.focus())
    } catch (error) {
      setError((error as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="page">
      <PageHeading number="01" title="Define your reference.">
        Choose the population context for your first synthetic cohort.
      </PageHeading>
      <Disclaimer />
      <div className="workspace-intro">
        <span className="eyebrow">The starting point</span>
        <p>
          Every comparison begins with a complete profile. Select a reference below, then adjust its
          eight conditions.
        </p>
      </div>
      <ProfileForm initial={state.baseline} busy={busy} onSubmit={generate} />
      {error && (
        <ErrorBox
          message={error}
          onRetry={() => {
            if (submitted) void generate(submitted)
          }}
        />
      )}
      {result && (
        <section className="results-section" id="results">
          <div className="results-heading">
            <div>
              <span className="eyebrow">
                {busy || error ? 'Last successful result' : 'Your reference / results'}
              </span>
              <h2 ref={resultHeading} tabIndex={-1}>
                Synthetic Cohort Generated
              </h2>
            </div>
            <span className="result-count">{result.outcome_stat.n.toLocaleString()} profiles</span>
          </div>
          <MetricCard result={result} />
          <div className="result-detail-grid">
            <div className="detail-card">
              <h3>Effective conditions</h3>
              <ConditionDetails condition={result.effective_conditions} />
            </div>
            <div className="detail-card">
              <span className="eyebrow">Reading this cohort</span>
              <h3>Context matters.</h3>
              <p>{result.note}</p>
              <p>
                <b>Scope</b>
                <br />
                {result.scope}
              </p>
              <p>{result.disclaimer}</p>
              <details className="sampling-details">
                <summary>Generation diagnostics</summary>
                <p>
                  Rejection share:{' '}
                  {typeof result.sampling_diagnostics.rejection_rate === 'number'
                    ? `${(result.sampling_diagnostics.rejection_rate * 100).toFixed(1)}%`
                    : 'Not reported'}
                </p>
                <p>
                  Clipped share:{' '}
                  {typeof result.sampling_diagnostics.clipped_share === 'number'
                    ? `${(result.sampling_diagnostics.clipped_share * 100).toFixed(1)}%`
                    : 'Not reported'}
                </p>
                {typeof result.sampling_diagnostics.warning === 'string' && (
                  <p>{result.sampling_diagnostics.warning}</p>
                )}
              </details>
            </div>
          </div>
          <div className="section-title">
            <span className="eyebrow">A closer look</span>
            <h2>Representative examples</h2>
            <p>
              A few generated profiles to make the cohort tangible. The full cohort determines the
              statistic.
            </p>
          </div>
          <Examples result={result} />
          <div className="next-step">
            <div>
              <span className="eyebrow">Next in your notebook</span>
              <h3>What changes with the context?</h3>
              <p>Keep this reference and explore a second set of conditions.</p>
            </div>
            <Link to="/compare" className="button primary">
              Compare a scenario <ArrowRight size={16} />
            </Link>
          </div>
        </section>
      )}
    </div>
  )
}
