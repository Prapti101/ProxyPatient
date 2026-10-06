import { useEffect, useId, useState } from 'react'
import { ArrowRight, Check, ChevronDown, SlidersHorizontal } from 'lucide-react'
import type { Condition, GenerateResponse, OptionsResponse, Profile } from '../../types/api'
import api, { dataMode } from '../../services/api'
import { ErrorBox, Loading, StatusBadge } from '../ui/Blocks'

const required = [
  'sex',
  'age_band',
  'residence',
  'wealth_quintile',
  'bmi_band',
  'hypertension',
  'tobacco',
  'alcohol',
] as const
const titles: Record<keyof Condition, string> = {
  sex: 'Sex',
  age_band: 'Age band',
  residence: 'Residence',
  wealth_quintile: 'Wealth quintile',
  bmi_band: 'BMI band',
  hypertension: 'Hypertension',
  tobacco: 'Tobacco use',
  alcohol: 'Alcohol use',
  state: 'State',
}
export function conditionText(key: string, value: unknown) {
  if (value === undefined || value === null || value === '') return 'Not selected'
  if (key === 'sex') return Number(value) === 0 ? 'Female' : 'Male'
  if (key === 'wealth_quintile') return `Quintile ${value}`
  if (['hypertension', 'tobacco', 'alcohol'].includes(key)) return Number(value) ? 'Yes' : 'No'
  if (key === 'age_band') return `${value} years`
  return String(value).replace(/^./, (letter) => letter.toUpperCase())
}
export function conditionSummary(condition?: Condition | null) {
  if (!condition) return 'No profile selected'
  return (
    [
      'sex',
      'age_band',
      'residence',
      'bmi_band',
      'wealth_quintile',
      'hypertension',
      'tobacco',
      'alcohol',
      'state',
    ] as const
  )
    .filter((key) => condition[key] !== undefined && condition[key] !== null)
    .map((key) => `${titles[key]}: ${conditionText(key, condition[key])}`)
    .join(' · ')
}
export function ConditionDetails({ condition }: { condition: Condition }) {
  return (
    <dl className="condition-details">
      {Object.entries(titles)
        .filter(([key]) => condition[key as keyof Condition] != null)
        .map(([key, label]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>{conditionText(key, condition[key as keyof Condition])}</dd>
          </div>
        ))}
    </dl>
  )
}

export function ProfileForm({
  initial,
  onSubmit,
  submitLabel = 'Generate baseline',
  busy = false,
}: {
  initial?: Condition | null
  onSubmit: (condition: Condition) => void
  submitLabel?: string
  busy?: boolean
}) {
  const [options, setOptions] = useState<OptionsResponse | null>(null)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [condition, setCondition] = useState<Condition>(initial ?? {})
  const id = useId()
  useEffect(() => {
    let active = true
    setError('')
    Promise.allSettled([api.getOptions(), api.getProfiles()]).then(
      ([optionsResult, profileResult]) => {
        if (!active) return
        if (optionsResult.status === 'fulfilled') setOptions(optionsResult.value)
        else
          setError(
            optionsResult.reason instanceof Error
              ? optionsResult.reason.message
              : 'Supported options could not be loaded.',
          )
        if (profileResult.status === 'fulfilled') {
          setProfiles(profileResult.value.profiles)
          setNotice('')
        } else
          setNotice(
            'Reference presets are unavailable. You can still select all eight conditions below.',
          )
      },
    )
    return () => {
      active = false
    }
  }, [attempt])
  useEffect(() => {
    if (initial) setCondition(initial)
  }, [initial])

  if (!options)
    return error ? (
      <ErrorBox message={error} onRetry={() => setAttempt((value) => value + 1)} />
    ) : (
      <Loading label="Loading profile options…" />
    )
  const fieldOptions: Record<keyof Condition, (string | number)[]> = {
    sex: options.sex,
    age_band: condition.sex === 1 ? options.age_band.men_options : options.age_band.women_options,
    residence: options.residence,
    wealth_quintile: options.wealth_quintile,
    bmi_band: options.bmi_band,
    hypertension: options.hypertension,
    tobacco: options.tobacco,
    alcohol: options.alcohol,
    state: options.state,
  }
  const complete = required.filter((key) =>
    fieldOptions[key].some((value) => value === condition[key]),
  ).length
  const valid =
    complete === required.length &&
    (condition.state == null || options.state.includes(condition.state))
  function change(key: keyof Condition, value: string) {
    setCondition((previous) => {
      const next = {
        ...previous,
        [key]:
          value === ''
            ? undefined
            : ['sex', 'wealth_quintile', 'hypertension', 'tobacco', 'alcohol', 'state'].includes(
                  key,
                )
              ? Number(value)
              : value,
      }
      if (key === 'sex') {
        const ages =
          Number(value) === 1 ? options!.age_band.men_options : options!.age_band.women_options
        if (next.age_band && !ages.includes(next.age_band)) next.age_band = undefined
      }
      return next
    })
  }
  const field = (key: keyof Condition) => (
    <label className="field" key={key}>
      <span>
        {titles[key]}
        {key === 'state' && <em>Optional</em>}
      </span>
      <div className="select-wrap">
        <select
          required={key !== 'state'}
          disabled={key === 'age_band' && condition.sex === undefined}
          value={condition[key] == null ? '' : String(condition[key])}
          onChange={(event) => change(key, event.target.value)}
        >
          <option value="">
            {key === 'state'
              ? 'Any supported state'
              : key === 'age_band' && condition.sex === undefined
                ? 'Select sex first'
                : `Select ${titles[key].toLowerCase()}`}
          </option>
          {fieldOptions[key].map((value) => (
            <option value={value} key={value}>
              {conditionText(key, value)}
            </option>
          ))}
        </select>
        <ChevronDown size={15} />
      </div>
    </label>
  )

  return (
    <form
      className="builder-card"
      onSubmit={(event) => {
        event.preventDefault()
        if (valid && !busy) onSubmit({ ...condition })
      }}
      aria-busy={busy}
      aria-describedby={`${id}-note`}
    >
      <fieldset disabled={busy} className="form-controls">
        <legend className="sr-only">Population profile</legend>
        <div className="preset-heading">
          <div>
            <span className="eyebrow">Start with a reference</span>
            <h3>{dataMode === 'demo' ? 'Illustrative profiles' : 'Supported profiles'}</h3>
          </div>
          <SlidersHorizontal size={19} />
        </div>
        <p className="muted">
          {dataMode === 'demo'
            ? 'Sample inputs for this interface demo. Choose one to fill the form.'
            : 'Choose a supported combination, then adjust the conditions below.'}
        </p>
        {notice && <p className="notice">{notice}</p>}
        {profiles.length > 0 && (
          <div className="preset-row">
            {profiles.map((profile, index) => {
              const selected = [...required, 'state' as const].every(
                (key) => (condition[key] ?? null) === (profile.condition[key] ?? null),
              )
              return (
                <button
                  type="button"
                  className={`preset ${selected ? 'selected' : ''}`}
                  aria-pressed={selected}
                  key={`${profile.label}-${index}`}
                  onClick={() => setCondition({ ...profile.condition })}
                >
                  <span className="preset-index">0{index + 1}</span>
                  <span>
                    <b>{profile.label}</b>
                    <small>{profile.description}</small>
                  </span>
                  {selected ? <Check size={16} /> : <ArrowRight size={16} />}
                </button>
              )
            })}
          </div>
        )}
        <div className="form-section">
          <div className="form-section-heading">
            <span>01</span>
            <h3>Population context</h3>
            <small>All four fields required</small>
          </div>
          <div className="field-grid">
            {(['sex', 'age_band', 'residence', 'wealth_quintile'] as const).map(field)}
          </div>
        </div>
        <div className="form-section">
          <div className="form-section-heading">
            <span>02</span>
            <h3>Health & lifestyle</h3>
            <small>All four fields required</small>
          </div>
          <div className="field-grid">
            {(['bmi_band', 'hypertension', 'tobacco', 'alcohol'] as const).map(field)}
          </div>
        </div>
        {options.state.length > 0 && (
          <div className="optional-field">
            {field('state')}
            <p>
              The model uses supported raw state codes. Leaving this blank allows supported states
              to vary.
            </p>
          </div>
        )}
      </fieldset>
      <div className="form-bottom">
        <span className={valid ? 'valid-note' : 'hint-note'} aria-live="polite">
          {valid ? (
            <>
              <Check size={16} />
              Profile ready
            </>
          ) : (
            <>
              <b>{complete} / 8</b> required conditions selected
            </>
          )}
        </span>
        <button className="button primary" disabled={!valid || busy}>
          {busy ? (
            <>
              <span className="spinner" />
              Generating…
            </>
          ) : (
            <>
              {submitLabel}
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
      <p className="micro-note" id={`${id}-note`}>
        Glucose is the generated outcome, never an input. Reference profiles represent combinations
        of conditions, not real people.
      </p>
      {busy && (
        <p className="request-status" role="status">
          Generating the cohort. Results appear when the request completes.
        </p>
      )}
    </form>
  )
}

export function MetricCard({
  result,
  title = 'Generated cohort',
}: {
  result: GenerateResponse
  title?: string
}) {
  const outcome = result.outcome_stat
  const reportedLevel = result.outcome_stat.monte_carlo_interval?.level
  const level = typeof reportedLevel === 'number' ? Math.round(reportedLevel * 100) : 95
  return (
    <section className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <StatusBadge
          status={result.status_banner}
          runType={result.run_type}
          preliminary={result.preliminary}
          demo={result.demo}
        />
      </div>
      <div className="metric-primary">
        <div className="metric-value">
          {outcome.rate_pct.toFixed(1)}
          <small>%</small>
        </div>
        <div className="metric-label">
          Elevated glucose <span>(proxy)</span>
        </div>
      </div>
      <div className="metric-bottom">
        <div>
          <small>{level}% interval</small>
          <b>
            {(outcome.ci_low * 100).toFixed(1)}–{(outcome.ci_high * 100).toFixed(1)}%
          </b>
        </div>
        <div>
          <small>Synthetic cohort</small>
          <b>{outcome.n.toLocaleString()} profiles</b>
        </div>
        <div>
          <small>Weighting</small>
          <b>{result.weighting}</b>
        </div>
      </div>
      {result.run_note && <p className="run-note">{result.run_note}</p>}
      <p className="metric-footnote">{result.uncertainty_note}</p>
    </section>
  )
}
export function Examples({ result }: { result: GenerateResponse }) {
  return (
    <div className="examples-section">
      {!result.examples?.length ? (
        <div className="notice">
          <b>Examples withheld</b>
          <p>{result.examples_status}. The cohort summary remains available.</p>
        </div>
      ) : (
        <div className="example-grid">
          {result.examples.slice(0, 5).map((example) => {
            const features = example.generated_features
            const values: [string, unknown, string][] = [
              ['Age', features.age, 'years'],
              ['BMI', features.bmi, 'kg/m²'],
              ['Waist', features.waist_cm, 'cm'],
              ['Hip', features.hip_cm, 'cm'],
              ['Weight', features.weight_kg, 'kg'],
              ['Height', features.height_cm, 'cm'],
              ['Education code', features.education, ''],
              [
                'BP checked',
                features.bp_ever_checked == null
                  ? null
                  : Number(features.bp_ever_checked)
                    ? 'Yes'
                    : 'No',
                '',
              ],
            ]
            return (
              <article className="example-card" key={example.example_id}>
                <div className="example-head">
                  <span>{example.example_id}</span>
                  <span className="record-label">
                    {result.demo ? 'MOCK / SYNTHETIC' : 'SYNTHETIC'}
                  </span>
                </div>
                <p className="example-warning">{example.label}</p>
                <dl className="feature-grid">
                  {values
                    .filter(([, value]) => value != null)
                    .map(([key, value, unit]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>
                          {typeof value === 'number'
                            ? value.toFixed(key === 'Age' || key === 'Education code' ? 0 : 1)
                            : String(value)}
                          {unit && <small> {unit}</small>}
                        </dd>
                      </div>
                    ))}
                </dl>
                <div className="glucose-row">
                  <span>Generated glucose</span>
                  <b>
                    {example.glucose_raw} <small>mg/dL</small>
                  </b>
                  <span className={example.elevated_glucose_proxy ? 'elevated' : 'not-elevated'}>
                    {example.elevated_glucose_proxy
                      ? 'Elevated glucose (proxy)'
                      : 'Below the elevated glucose (proxy) threshold'}
                  </span>
                </div>
                {example.illustrative_elevated && (
                  <p className="illustration-note">
                    Included to illustrate an elevated glucose (proxy) outcome.
                  </p>
                )}
                <details className="example-conditions">
                  <summary>Effective conditions</summary>
                  <ConditionDetails condition={example.effective_conditions} />
                </details>
              </article>
            )
          })}
        </div>
      )}
      <p className="examples-note">{result.examples_note}</p>
    </div>
  )
}
