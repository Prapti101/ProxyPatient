import { useEffect, useState } from 'react'
import { ArrowDown, ArrowRight, ArrowUpRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import api, { dataMode } from '../services/api'
import type { HealthResponse } from '../types/api'
import { Loading, StatusBadge } from '../components/ui/Blocks'

const chapters = [
  {
    number: '01',
    title: 'Define the context',
    text: 'Choose eight population conditions. Start with a reference profile, then make it your own.',
    to: '/explore',
    action: 'Build a profile',
  },
  {
    number: '02',
    title: 'Read the difference',
    text: 'Compare generated groups side by side, with intervals and a shared reference.',
    to: '/compare',
    action: 'Compare scenarios',
  },
  {
    number: '03',
    title: 'Look at the evidence',
    text: 'Read the measured reports and their scope. An absent report stays visibly pending.',
    to: '/validation',
    action: 'Review validation',
  },
]

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    api
      .health()
      .then((value) => {
        if (active) setHealth(value)
      })
      .catch((e) => {
        if (active) setError(e.message)
      })
    return () => {
      active = false
    }
  }, [])
  return (
    <div className="page home-page">
      <div className="edition-line">
        <span>FIELDNOTES / SYNTHETIC POPULATIONS</span>
        <span>NFHS-5 · INDIA</span>
      </div>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">A workspace for asking better questions</span>
          <h1>
            Change the context.
            <br />
            <em>Explore the patterns.</em>
          </h1>
          <p>
            Build synthetic populations and explore their elevated glucose (proxy) outcomes. One
            reference profile, a considered comparison, and the evidence to read it.
          </p>
          <div className="hero-actions">
            <Link to="/explore" className="button primary">
              Build a reference profile <ArrowRight size={17} />
            </Link>
            <Link to="/how-it-works" className="text-link">
              Read the method <ArrowUpRight size={15} />
            </Link>
          </div>
          <div className="hero-status">
            {health ? (
              <StatusBadge
                status={health.status_banner}
                runType={health.run_type}
                preliminary={health.preliminary}
                demo={health.demo}
              />
            ) : error ? (
              <span className="status-badge unknown">Status unavailable</span>
            ) : (
              <Loading label="Checking model status…" />
            )}
            <span>
              {error
                ? 'The service could not be reached.'
                : !health
                  ? 'Connecting to the current session.'
                  : health.status !== 'ok' || health.mode === 'unavailable'
                    ? 'Model unavailable for generation.'
                    : dataMode === 'demo'
                      ? 'Illustrative interface data.'
                      : health.demo
                        ? 'Mock-trained demonstration model.'
                        : 'See run scope before interpreting results.'}
            </span>
          </div>
        </div>
        <figure className="process-plate">
          <figcaption>
            <span>THE SCENARIO PROCESS</span>
            <span>FIG. 01</span>
          </figcaption>
          <div className="plate-input">
            <span className="plate-step">01 / SET CONDITIONS</span>
            <h2>A population profile</h2>
            <div className="condition-tokens">
              <span>Age band</span>
              <span>BMI band</span>
              <span>Residence</span>
              <span>+ 5 conditions</span>
            </div>
          </div>
          <div className="plate-bridge">
            <span />
            <ArrowDown size={17} />
            <span />
          </div>
          <div className="plate-model">
            <span className="model-symbol" aria-hidden="true">
              c<span>VAE</span>
            </span>
            <div>
              <b>Conditional generation</b>
              <p>Learned patterns → new profiles</p>
            </div>
          </div>
          <div className="plate-bridge">
            <span />
            <ArrowDown size={17} />
            <span />
          </div>
          <div className="plate-output">
            <div className="cohort-glyph" aria-hidden="true">
              {Array.from({ length: 40 }, (_, i) => (
                <i key={i} />
              ))}
            </div>
            <div>
              <span className="plate-step">02 / READ THE COHORT</span>
              <h3>A group-level summary</h3>
              <p>Outcome · interval · examples</p>
            </div>
          </div>
          <p className="plate-caption">Process illustration. No measured values shown.</p>
        </figure>
      </section>
      <div className="scope-note">
        <span className="eyebrow">A note on interpretation</span>
        <p>
          Generated profiles are synthetic. Comparisons describe groups; they do not establish
          causes or give a medical diagnosis.
        </p>
      </div>
      <section className="chapters-section" aria-labelledby="chapters-heading">
        <div className="section-title">
          <span className="eyebrow">A considered workflow</span>
          <h2 id="chapters-heading">From a question to its context.</h2>
        </div>
        <div className="chapter-grid">
          {chapters.map((chapter) => (
            <Link className="chapter" to={chapter.to} key={chapter.number}>
              <span className="chapter-number">{chapter.number}</span>
              <h3>{chapter.title}</h3>
              <p>{chapter.text}</p>
              <span className="chapter-action">
                {chapter.action}
                <ArrowUpRight size={17} />
              </span>
            </Link>
          ))}
        </div>
      </section>
      <section className="research-note">
        <span className="eyebrow">The research context</span>
        <h2>
          Modelled populations.
          <br />
          Human interpretation.
        </h2>
        <div>
          <p>
            The main model is a conditional VAE designed for NFHS-5 India survey data. Its output is
            a synthetic cohort, summarised for elevated glucose (proxy).
          </p>
          <p>
            A run label tells you whether you are looking at mock data, a preliminary quick run, or
            a full run. Validation reports explain what has actually been measured.
          </p>
          <Link to="/how-it-works" className="text-link">
            Understand the scope <ArrowRight size={15} />
          </Link>
        </div>
      </section>
    </div>
  )
}
