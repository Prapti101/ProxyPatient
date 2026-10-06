import { useRef, useState } from 'react'
import { ArrowDownRight, ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAppState } from '../context/AppState'
import { ConditionDetails, Examples, MetricCard } from '../components/scenarios/ScenarioUI'
import { Empty, ExploreLink, PageHeading, StatusBadge } from '../components/ui/Blocks'

export default function Scenarios() {
  const { history } = useAppState()
  const [selected, setSelected] = useState<number | null>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  function select(index: number) {
    setSelected(index)
    requestAnimationFrame(() => heading.current?.focus())
  }
  return (
    <div className="page">
      <PageHeading number="03" title="Your scenario notebook.">
        A record of this session, in the order you generated it.
      </PageHeading>
      <div className="notebook-meta">
        <span>
          {history.length} {history.length === 1 ? 'entry' : 'entries'}
        </span>
        <p>Session only · refreshing the page clears these entries.</p>
      </div>
      {history.length === 0 ? (
        <Empty title="A fresh page for your first question" action={<ExploreLink />}>
          Generate a reference cohort to start your notebook. Each successful result will appear
          here.
        </Empty>
      ) : (
        <>
          <div className="history-list">
            {history.map((result, index) => (
              <button
                key={index}
                className={`history-entry ${selected === index ? 'selected' : ''}`}
                aria-pressed={selected === index}
                onClick={() => select(index)}
                aria-label={`Open entry ${index + 1}, ${result.outcome_stat.rate_pct.toFixed(1)} percent elevated glucose proxy`}
              >
                <span className="entry-number">{String(index + 1).padStart(2, '0')}</span>
                <div className="entry-description">
                  <span className="entry-title">
                    {'label' in result ? String(result.label) : 'Reference cohort'}
                  </span>
                  <p>
                    {result.effective_conditions.sex === 0 ? 'Female' : 'Male'} ·{' '}
                    {result.effective_conditions.age_band} years ·{' '}
                    {result.effective_conditions.residence} · {result.effective_conditions.bmi_band}{' '}
                    BMI
                  </p>
                  <StatusBadge
                    status={result.status_banner}
                    runType={result.run_type}
                    preliminary={result.preliminary}
                    demo={result.demo}
                  />
                </div>
                <div className="history-metric">
                  <b>
                    {result.outcome_stat.rate_pct.toFixed(1)}
                    <small>%</small>
                  </b>
                  <span>elevated glucose (proxy)</span>
                  <small>{result.outcome_stat.n.toLocaleString()} synthetic profiles</small>
                </div>
                <ArrowDownRight size={20} />
              </button>
            ))}
          </div>
          {selected !== null && history[selected] && (
            <section className="results-section selected-scenario">
              <div className="results-heading">
                <div>
                  <span className="eyebrow">From your session</span>
                  <h2 ref={heading} tabIndex={-1}>
                    Entry {String(selected + 1).padStart(2, '0')}
                  </h2>
                </div>
                <Link to="/explore" className="text-link">
                  Create another <ArrowRight size={15} />
                </Link>
              </div>
              <MetricCard result={history[selected]} />
              <div className="detail-card notebook-conditions">
                <h3>Effective conditions</h3>
                <ConditionDetails condition={history[selected].effective_conditions} />
              </div>
              <Examples result={history[selected]} />
            </section>
          )}
        </>
      )}
    </div>
  )
}
