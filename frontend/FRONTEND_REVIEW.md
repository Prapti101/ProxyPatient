# Frontend review and design refinement

6 October 2026 · Reviewed against fork main `941cfcf` (including P4's merged frontend). Work branch: `p4-frontend-refinement`.

## Design direction

The interface is now a **population research notebook**: warm paper, dark botanical ink, one restrained green accent and a muted rust color for the second comparison cohort. Serif headings distinguish narrative from data; system sans-serif text and tabular numbers keep controls and results legible. Small monospace labels establish the sequence without dominating it.

The home page contains a deliberately drawn process figure with no invented measurements. Ruled sections replace repeated promotional cards. Results lead with the cohort statistic, then its interval, run provenance and conditions. Notebook entries are chronological. Validation treats pending evidence as a normal, clearly explained state.

The shared stylesheet replaces the original dark theme plus partial light-theme overrides. No remote font request or decorative orbit animation remains. Existing dependencies and lockfile are unchanged; removing page-motion imports from the runtime also reduced the production entry JavaScript from 287.03 KB to approximately 182.49 KB before gzip. The comparison chart remains separately lazy-loaded.

## Code review findings addressed

| Priority | Finding and user impact | Resolution |
|---|---|---|
| High | Explore updated the shared baseline before generation succeeded. A failed request could leave a new profile paired with an old outcome, affecting the next comparison. | Accept the baseline and its result together only after a successful response. Failed attempts retain the successful reference and expose a correctly targeted retry. |
| High | Previous comparison results remained after replacing the reference profile. | Clear the comparison when accepting a new baseline. Read results from shared state rather than maintaining a second stale copy. |
| High | Some result panels omitted run provenance, preliminary notes and example-screening limitations. Demo values could look like measured model outputs. | Persistent interface-demo context, per-result run badges, returned run/uncertainty notes and the API's examples_note. Null examples remain withheld while the cohort statistic stays visible. |
| Medium | Validation and model comparison were loaded with Promise.all and comparison was hidden behind the validation status. One unavailable report suppressed the other. | Load independently with Promise.allSettled. Each panel presents its own status, error, metrics and provenance. |
| Medium | Retrying unavailable options reloaded the entire app and erased session history. | Retry the option request in place; preserve the notebook and successful reference. Missing presets still permit manual input. |
| Medium | Inputs and presets remained editable during generation, making the submitted conditions hard to distinguish from changes made while waiting. | Disable the form while a request is pending; show an indeterminate request message. No fabricated progress percentage. |
| Medium | Session entries were drawn as branches of one reference without storing those relationships. | Replace the graph with an ordered notebook. Entries can be inspected without implying unsupported lineage. |
| Medium | Condition summaries omitted tobacco, alcohol and optional state. | Include all effective conditions in a readable definition list; examples also expose their returned conditions. |
| Medium | The comparison chart computed interval values but did not render them. | Show error bars, consistent reference/what-if colors, zero-based percentage axes and a readable text description alongside the cards. |
| Medium | Mixed theme rules left dark surfaces/light text in parts of the otherwise light interface; many labels were 7–10px. | One coherent palette, clear hierarchy, larger controls and primary text, wrapped metadata and responsive grids. |
| Medium | Navigation lacked expanded-state semantics, keyboard dismissal, route focus management and a skip link. | Added aria-expanded/aria-controls, Escape handling, visible focus, skip-to-content, route titles and focus/scroll management. Reduced-motion preferences are respected. |
| Low | The home page could say the status check failed while it was still loading. | Distinct connecting, unavailable, interface-demo, mock-model and real-mode descriptions. |
| Low | Unknown routes silently displayed the home page. | Show an explicit missing-page state with a route back to the workspace. |

During automated accessibility checks, low-contrast decorative page numbers and preset indices were identified and darkened. No checks were disabled to pass the audit.

## Findings intentionally left outside this change

- **Existing local demo fixtures are illustrations, not model inference.** `src/mocks/demoData.ts` contains a hand-written rate formula and example measurements that do not necessarily match every selected age/BMI band. Its generator also does not implement the complete n/seed/n_examples request contract. The current UI submits its existing fixed 1,000/three-example requests. The persistent interface-demo label makes the distinction explicit; the fixture model was not rewritten as part of a visual refinement. Use `VITE_DATA_MODE=api` for model-backed requests. Never present default demo values as measured research results.
- **Root documentation is inconsistent after the P4 merge.** The opening README still says a frontend is absent, while its later section documents the frontend. No root documentation was edited under the frontend-only instruction.
- **Root P3 copies differ from the active implementation.** `validation_report.json` still describes bootstrap intervals and different suppression/distance behavior; root `parser.py` describes a BERT/glucose-text path. The served endpoints use `backend/parser.py`, `backend/reports.py` and the guarded backend contract instead. Root `model_comparison.json` is an architecture table with null metrics, not a measured report. These files were reviewed but not changed or wired into the client.
- **Recharts 2.x is deprecated upstream.** npm reports this existing dependency warning. No dependency migration was mixed into this change.

These are review observations, not claims of a new real-data training run or full scientific validation. No backend, model, data pipeline, root configuration, existing API transport, or mock-data computation was changed.

## Verification

- Production build: `npm run build` passes, including TypeScript checks.
- Nine browser regression tests pass in Chromium. They cover the demo generation/compare/notebook journey, clearing comparisons after a new baseline, API failure and retry without changing the reference, age-band validation, independent reports, suppressed values, withheld examples and quick labels, option retry without losing history, pending controls, keyboard/mobile navigation, unavailable health and missing routes.
- Responsive checks cover all six routes at 320, 390, 768, 1024 and 1440 pixels. Populated generation and comparison views are also exercised on mobile. No horizontal document overflow was observed.
- Axe WCAG A/AA checks: zero reported violations across six routes, generated results, comparison results and mobile navigation. This is automated coverage, not a claim of complete accessibility certification.
- Actual FastAPI smoke: the unchanged backend served existing mock-trained artifacts. The browser loaded presets, generated three labelled examples, displayed the returned rate, compared cohorts, displayed pending reports and retained three notebook entries. Private survey data was not used.
- Desktop/mobile screenshots were inspected. Artifacts are outside the checkout at `/workspace/artifacts/frontend-review/`.
- Scope check: every tracked file changed by this work is under `frontend/`. The pre-existing untracked `docs/PROJECT_PROGRESS_OVERVIEW.md` is preserved.

## Reproduce the checks

From `frontend/`:

```bash
npm ci
npm run build
npm run dev -- --host 127.0.0.1 --port 4173
```

In a second terminal, also from `frontend/`:

```bash
VITE_DATA_MODE=api npm run dev -- --host 127.0.0.1 --port 4174
```

The browser tests require Python Playwright and a Chromium executable. They intercept API-mode network requests with constructed test fixtures and do not require the backend. From the repository root:

```bash
# Install these test-only tools into your chosen local test environment if absent.
python -m pip install playwright
# Set CHROMIUM_PATH if Chromium is not at /usr/bin/chromium.
python frontend/tests/browser_checks.py

# Axe is a test-only tool installed outside the repository dependency tree.
npm install --prefix /tmp/proxypatient-a11y --no-audit --no-fund axe-core
AXE_SCRIPT=/tmp/proxypatient-a11y/node_modules/axe-core/axe.min.js \
  python frontend/tests/accessibility_checks.py
```

`PP_FRONTEND_DEMO_URL` and `PP_FRONTEND_API_URL` override the Vite URLs. Use the defaults in `.env.example` for the documented intercepted API origin (`http://localhost:8000`). The live FastAPI smoke was separate from these fixture-based regression checks.

### Theme consistency follow-up

The notebook uses one light palette. Explicit `only light` color-scheme declarations keep native browser controls consistent when the operating system prefers dark mode. The body has an explicit paper background, and leftover Tailwind dark palette tokens now match the notebook colors. Deep green primary buttons remain intentional accents. A browser regression compares all six routes and their controls under light and dark preferences.
