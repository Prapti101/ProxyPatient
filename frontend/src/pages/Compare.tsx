import { useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import api from '../services/api'
import { useAppState } from '../context/AppState'
import {
  ConditionDetails,
  Examples,
  MetricCard,
  ProfileForm,
} from '../components/scenarios/ScenarioUI'
import {
  Disclaimer,
  Empty,
  ErrorBox,
  ExploreLink,
  PageHeading,
  SectionTitle,
} from '../components/ui/Blocks'
import type { Condition } from '../types/api'

export default function Compare() {
  const state = useAppState()
  const [whatif, setWhatif] = useState<Condition | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)
  const results = state.comparison
  async function compare(condition: Condition) {
    if (!state.baseline) return
    setWhatif(condition)
    setError('')
    setBusy(true)
    try {
      const response = await api.compare({
        scenarios: [
          { label: 'Reference profile', condition: state.baseline, n: 1000, n_examples: 3 },
          { label: 'What-if scenario', condition, n: 1000, n_examples: 3 },
        ],
      })
      state.setComparison(response.scenarios)
      response.scenarios.forEach((result) => state.addHistory(result))
      requestAnimationFrame(() => heading.current?.focus())
    } catch (error) {
      setError((error as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const chartData = results.slice(0, 2).map((result) => ({
    name: result.label,
    rate: result.outcome_stat.rate_pct,
    interval: [
      (result.outcome_stat.rate - result.outcome_stat.ci_low) * 100,
      (result.outcome_stat.ci_high - result.outcome_stat.rate) * 100,
    ],
  }))
  return (
    <div className="page">
      <PageHeading number="02" title="Change the context.">
        Keep a reference in view. Compare it with a second synthetic population.
      </PageHeading>
      <Disclaimer />
      {!state.baseline ? (
        <Empty title="Begin with a reference profile" action={<ExploreLink />}>
          Generate your first cohort in Explore. It becomes the reference for this comparison.
        </Empty>
      ) : (
        <>
          <div className="baseline-reference">
            <div>
              <span className="eyebrow">Your reference / held for comparison</span>
              <h2>The starting conditions</h2>
            </div>
            <ConditionDetails condition={state.baseline} />
          </div>
          <SectionTitle eyebrow="The second scenario" title="What would you change?">
            Adjust any condition below. A new request generates both cohorts together.
          </SectionTitle>
          <ProfileForm
            initial={whatif ?? state.baseline}
            submitLabel="Compare scenarios"
            busy={busy}
            onSubmit={compare}
          />
          {error && (
            <ErrorBox
              message={error}
              onRetry={() => {
                if (whatif) void compare(whatif)
              }}
            />
          )}
        </>
      )}
      {results.length > 0 && (
        <section className="results-section">
          <div className="results-heading">
            <div>
              <span className="eyebrow">
                {busy || error ? 'Last successful comparison' : 'The comparison / results'}
              </span>
              <h2 ref={heading} tabIndex={-1}>
                Two contexts, side by side.
              </h2>
            </div>
          </div>
          <p className="causal-note">
            These differences describe generated groups. They do not demonstrate that changing a
            condition causes an outcome.
          </p>
          <div className="compare-grid">
            {results.slice(0, 2).map((result, index) => (
              <article
                key={`${result.label}-${index}`}
                className={`comparison-column column-${index}`}
              >
                <div className="comparison-label">
                  <span>{index === 0 ? 'A' : 'B'}</span>
                  <h3>{index === 0 ? 'Reference profile' : 'What-if scenario'}</h3>
                </div>
                <MetricCard result={result} title={result.label} />
                <details className="comparison-conditions">
                  <summary>View effective conditions</summary>
                  <ConditionDetails condition={result.effective_conditions} />
                </details>
              </article>
            ))}
          </div>
          {results.length >= 2 && (
            <div className="delta-card">
              <div>
                <span className="eyebrow">Change from reference</span>
                <b>
                  {results[1].delta_pp > 0 ? '+' : ''}
                  {results[1].delta_pp.toFixed(2)} <small>pp</small>
                </b>
                <p>
                  Percentage points, not percent change.
                  <br />A descriptive difference between cohorts.
                </p>
              </div>
              <div
                className="chart-wrap"
                role="img"
                aria-label={`Reference ${results[0].outcome_stat.rate_pct.toFixed(2)} percent; what-if ${results[1].outcome_stat.rate_pct.toFixed(2)} percent. Intervals are listed in the result cards.`}
              >
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart
                    accessibilityLayer
                    data={chartData}
                    margin={{ top: 15, right: 16, bottom: 6, left: 0 }}
                  >
                    <CartesianGrid vertical={false} strokeDasharray="3 3" stroke="#d9dfd8" />
                    <XAxis
                      dataKey="name"
                      tickLine={false}
                      axisLine={false}
                      stroke="#58665f"
                      fontSize={12}
                    />
                    <YAxis
                      unit="%"
                      tickLine={false}
                      axisLine={false}
                      stroke="#58665f"
                      fontSize={12}
                      domain={[
                        0,
                        (maximum: number) => Math.min(100, Math.max(1, Math.ceil(maximum * 1.15))),
                      ]}
                    />
                    <Tooltip
                      cursor={{ fill: '#edf0e9' }}
                      contentStyle={{
                        background: '#fffefb',
                        border: '1px solid #cbd3c9',
                        borderRadius: 4,
                        fontSize: 13,
                      }}
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                    />
                    <Bar
                      dataKey="rate"
                      name="Elevated glucose (proxy)"
                      maxBarSize={74}
                      radius={[3, 3, 0, 0]}
                      isAnimationActive={false}
                    >
                      {chartData.map((_, index) => (
                        <Cell key={index} fill={index ? '#a66b45' : '#246453'} />
                      ))}
                      <ErrorBar dataKey="interval" width={8} stroke="#34463f" strokeWidth={1.5} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          <SectionTitle eyebrow="Inside each cohort" title="Representative examples">
            Illustrations from each result. Read the labels and example availability alongside the
            numbers.
          </SectionTitle>
          <div className="compare-grid">
            {results.slice(0, 2).map((result, index) => (
              <section key={index}>
                <h3 className="examples-column-heading">
                  {index === 0 ? 'A / Reference' : 'B / What-if'}
                </h3>
                <Examples result={result} />
              </section>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
