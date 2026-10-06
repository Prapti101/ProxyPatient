import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Disclaimer, PageHeading } from '../components/ui/Blocks'

const steps = [
  [
    'The research context',
    'NFHS-5 India provides the survey context for model training. Private respondent records are never served by this interface.',
  ],
  [
    'Prepare the data',
    'Measurements, categories and supported conditions are checked before the training data is converted into model inputs.',
  ],
  [
    'Learn the patterns',
    'A conditional variational autoencoder (CVAE) learns relationships from the configured training data. The main model uses an MLP architecture.',
  ],
  [
    'Choose a reference',
    'Select eight population conditions, with an optional supported state. The complete profile becomes your starting point.',
  ],
  [
    'Ask a what-if',
    'Adjust the profile to describe a second population. It is a comparison of conditions, not a simulation of a person changing over time.',
  ],
  [
    'Generate the cohorts',
    'The trained model produces new synthetic profiles under each condition set. All displayed examples remain labelled synthetic.',
  ],
  [
    'Read the group outcome',
    'The full generated cohort determines the elevated glucose (proxy) statistic. Representative examples illustrate the result; they do not determine the percentage.',
  ],
  [
    'Inspect the evidence',
    'Compatible measured reports describe distributional fidelity, comparisons and limitations. Missing reports remain pending.',
  ],
]
export default function HowItWorks() {
  return (
    <div className="page">
      <PageHeading number="05" title="The method, in context.">
        From population conditions to synthetic cohorts, with interpretation at every step.
      </PageHeading>
      <Disclaimer />
      <div className="method-layout">
        <aside className="method-aside">
          <span className="eyebrow">Method notes</span>
          <h2>
            A model of patterns.
            <br />
            <em>A tool for questions.</em>
          </h2>
          <p>The distinction matters: a generated group is not an individual medical prediction.</p>
          <dl>
            <div>
              <dt>Main model</dt>
              <dd>Conditional VAE</dd>
            </div>
            <div>
              <dt>Research context</dt>
              <dd>NFHS-5 · India</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>Synthetic group summaries</dd>
            </div>
          </dl>
        </aside>
        <section className="pipeline" aria-label="Project workflow">
          {steps.map(([title, description], index) => (
            <article className="pipeline-step" key={title}>
              <span className="pipeline-number">{String(index + 1).padStart(2, '0')}</span>
              <div>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
            </article>
          ))}
        </section>
      </div>
      <section className="method-glossary">
        <div className="section-title">
          <span className="eyebrow">Reading the output</span>
          <h2>Three things to keep in view.</h2>
        </div>
        <div className="explain-grid">
          <article>
            <span>01 / THE OUTCOME</span>
            <h3>Elevated glucose (proxy)</h3>
            <p>
              The share of generated glucose readings at or above the configured threshold. It is a
              cohort statistic, not a diagnosis.
            </p>
          </article>
          <article>
            <span>02 / THE INTERVAL</span>
            <h3>Variation within the model</h3>
            <p>
              A Wilson Monte Carlo interval describes finite-cohort variation conditional on the
              fitted model. It excludes model and survey uncertainty. Interface-demo intervals are
              illustrative placeholders.
            </p>
          </article>
          <article>
            <span>03 / THE EXAMPLES</span>
            <h3>Illustrations with limits</h3>
            <p>
              Real-mode examples require a passing packaged nearest-record check. This is a
              heuristic, not a privacy guarantee. If examples are withheld, the group summary can
              still be available.
            </p>
          </article>
        </div>
      </section>
      <div className="next-step">
        <div>
          <span className="eyebrow">Put the method into practice</span>
          <h3>Start with a reference profile.</h3>
        </div>
        <Link to="/explore" className="button primary">
          Open the workspace <ArrowRight size={16} />
        </Link>
      </div>
    </div>
  )
}
