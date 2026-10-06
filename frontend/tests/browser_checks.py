"""Frontend-only browser regression checks; never reads survey data.

Start Vite on 4173 (demo) and 4174 (VITE_DATA_MODE=api), then run:
  python frontend/tests/browser_checks.py
Requires Python Playwright and Chromium; see FRONTEND_REVIEW.md.
"""
import copy
import json
import os
import unittest
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, expect

DEMO = os.environ.get('PP_FRONTEND_DEMO_URL', 'http://127.0.0.1:4173')
API = os.environ.get('PP_FRONTEND_API_URL', 'http://127.0.0.1:4174')


class FrontendChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=os.environ.get('CHROMIUM_PATH', '/usr/bin/chromium'),
            headless=True, args=['--no-sandbox'])
        page = cls.browser.new_page()
        page.goto(DEMO)
        cls.fixtures = page.evaluate("""async () => {
            const d = await import('/src/mocks/demoData.ts');
            return {options:d.options, profiles:d.profileResponse, health:d.health,
                    validation:d.validation, generated:d.demoGenerate(d.profiles[0].condition,'S1')};
        }""")
        page.close()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width':1440, 'height':1000})
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.posts = []
        self.responses = {
            '/health': (200, copy.deepcopy(self.fixtures['health'])),
            '/options': (200, copy.deepcopy(self.fixtures['options'])),
            '/profiles': (200, copy.deepcopy(self.fixtures['profiles'])),
            '/generate': (200, copy.deepcopy(self.fixtures['generated'])),
            '/validation': (200, copy.deepcopy(self.fixtures['validation'])),
            '/model-comparison': (200, copy.deepcopy(self.fixtures['validation'])),
        }

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], 'Browser runtime errors')

    def intercept(self):
        def route_request(route):
            path = urlparse(route.request.url).path
            if route.request.method == 'POST':
                self.posts.append((path, route.request.post_data_json))
            status, body = self.responses.get(path, (404, {'detail':'Unconfigured test route'}))
            route.fulfill(status=status, content_type='application/json', body=json.dumps(body),
                          headers={'Access-Control-Allow-Origin':'*', 'Access-Control-Allow-Headers':'*'})
        self.page.route('http://localhost:8000/**', route_request)

    def preset(self):
        self.page.get_by_role('button', name='Urban younger adult').click()

    def generate(self):
        self.preset()
        self.page.get_by_role('button', name='Generate baseline', exact=True).click()
        expect(self.page.get_by_role('heading', name='Synthetic Cohort Generated')).to_be_visible()

    def nav(self, name):
        if self.page.get_by_role('button', name='Open navigation').is_visible():
            self.page.get_by_role('button', name='Open navigation').click()
        self.page.get_by_role('navigation', name='Main navigation').get_by_role('link', name=name, exact=True).click()

    def assert_no_overflow(self):
        self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > innerWidth'),
                         f'Horizontal overflow: {self.page.url}')

    def test_light_theme_with_dark_browser_preference(self):
        snapshots = {}
        for preference in ('light', 'dark'):
            self.page.emulate_media(color_scheme=preference)
            for route in ('/', '/explore', '/compare', '/scenarios', '/validation', '/how-it-works'):
                self.page.goto(DEMO + route)
                self.page.wait_for_load_state('networkidle')
                expect(self.page.locator('main')).to_be_visible()
                styles = self.page.evaluate("""() => {
                    const style = el => {
                        const s = getComputedStyle(el);
                        return [s.backgroundColor, s.color, s.colorScheme];
                    };
                    return {root: style(document.documentElement), body: style(document.body),
                        controls: [...document.querySelectorAll('input, select, button')].map(style)};
                }""")
                self.assertIn('light', styles['root'][2])
                self.assertNotIn('dark', styles['root'][2])
                self.assertEqual(styles['body'][0], 'rgb(246, 245, 239)')
                if preference == 'light':
                    snapshots[route] = styles
                else:
                    self.assertEqual(styles, snapshots[route], route)

    def test_demo_journey_comparison_history_and_baseline_reset(self):
        self.page.goto(DEMO+'/explore')
        expect(self.page.get_by_role('button', name='Generate baseline', exact=True)).to_be_disabled()
        self.generate()
        expect(self.page.locator('.example-card')).to_have_count(3)
        expect(self.page.locator('.metric-card .status-badge')).to_contain_text('DEMO')
        self.nav('Compare')
        self.page.get_by_role('combobox', name='BMI band', exact=True).select_option('obese')
        self.page.get_by_role('button', name='Compare scenarios', exact=True).click()
        expect(self.page.get_by_role('heading', name='Two contexts, side by side.')).to_be_visible()
        expect(self.page.locator('.delta-card')).to_contain_text('+1.20')
        expect(self.page.locator('.recharts-errorBar')).to_have_count(2)
        self.nav('Notebook')
        expect(self.page.locator('.history-entry')).to_have_count(3)
        self.page.locator('.history-entry').nth(2).click()
        expect(self.page.get_by_role('heading', name='Entry 03', exact=True)).to_be_visible()
        self.nav('Explore')
        self.page.get_by_role('combobox', name='Residence', exact=True).select_option('rural')
        self.page.get_by_role('button', name='Generate baseline', exact=True).click()
        expect(self.page.locator('.results-section .condition-details').first).to_contain_text('Rural')
        self.nav('Compare')
        expect(self.page.get_by_role('heading', name='Two contexts, side by side.')).to_have_count(0)
        expect(self.page.locator('.baseline-reference')).to_contain_text('Rural')

    def test_api_failure_preserves_successful_baseline_and_retry(self):
        self.intercept()
        self.page.goto(API+'/explore')
        self.generate()
        self.responses['/generate'] = (503, {'detail':'Constructed unavailable model'})
        self.page.get_by_role('combobox', name='Residence', exact=True).select_option('rural')
        self.page.get_by_role('button', name='Generate baseline', exact=True).click()
        expect(self.page.get_by_role('alert')).to_contain_text('Constructed unavailable model')
        expect(self.page.locator('.results-heading')).to_contain_text('Last successful result')
        expect(self.page.locator('.results-section .condition-details').first).to_contain_text('Urban')
        self.page.get_by_role('button', name='Try again').click()
        expect(self.page.get_by_role('alert')).to_be_visible()
        self.assertEqual(self.posts[-1][1]['condition']['residence'], 'rural')
        self.nav('Compare')
        expect(self.page.locator('.baseline-reference')).to_contain_text('Urban')
        self.assertEqual(self.posts[0][1]['n_examples'], 3)
        self.assertNotIn('glucose', self.posts[0][1]['condition'])

    def test_age_options_clear_incompatible_band(self):
        self.page.goto(DEMO+'/explore')
        self.page.get_by_role('button', name='Rural adult').click()
        expect(self.page.get_by_role('combobox', name='Age band', exact=True)).to_have_value('35-54')
        self.page.get_by_role('combobox', name='Sex', exact=True).select_option('0')
        expect(self.page.get_by_role('combobox', name='Age band', exact=True)).to_have_value('')
        expect(self.page.get_by_role('button', name='Generate baseline', exact=True)).to_be_disabled()
        self.page.get_by_role('combobox', name='Age band', exact=True).select_option('35-49')
        expect(self.page.get_by_role('button', name='Generate baseline', exact=True)).to_be_enabled()

    def test_reports_render_independently_and_handle_suppression(self):
        self.intercept()
        self.responses['/validation'] = (503, {'detail':'Constructed validation failure'})
        self.responses['/model-comparison'] = (200, {
            'status':'complete', 'run_type':'quick', 'preliminary':True,
            'status_banner':'PRELIMINARY (quick run)', 'note':'Constructed report fixture, not a measured result',
            'models':{'CVAE':{'test_score':0.7, 'suppressed':None}},
            'evaluation_scope':{'split':'val', 'note':'fixture'}})
        self.page.goto(API+'/validation')
        expect(self.page.get_by_role('alert')).to_contain_text('Constructed validation failure')
        expect(self.page.locator('.report-panel').nth(1)).to_contain_text('0.7')
        expect(self.page.locator('.report-panel').nth(1)).to_contain_text('Not reported')
        self.responses['/validation'] = (200, self.fixtures['validation'])
        self.page.get_by_role('button', name='Refresh reports').click()
        expect(self.page.get_by_role('heading', name='The evidence is still pending.')).to_be_visible()
        expect(self.page.locator('.report-panel').nth(1)).to_contain_text('0.7')

    def test_quick_run_and_withheld_examples_remain_visible(self):
        self.intercept()
        result = copy.deepcopy(self.fixtures['generated'])
        result.update(demo=False, run_type='quick', preliminary=True,
                      status_banner='PRELIMINARY (quick run)', examples=None,
                      examples_status='withheld: near-copy check missing or not passed',
                      run_note='reduced, untuned run; results are indicative only',
                      examples_note='A heuristic, not a privacy guarantee.')
        self.responses['/generate'] = (200, result)
        self.page.goto(API+'/explore')
        self.generate()
        expect(self.page.locator('.metric-card')).to_contain_text('PRELIMINARY')
        expect(self.page.locator('.examples-section')).to_contain_text('near-copy check')
        expect(self.page.locator('.examples-section')).to_contain_text('not a privacy guarantee')
        expect(self.page.locator('.example-card')).to_have_count(0)
        expect(self.page.locator('.metric-value')).to_be_visible()

    def test_options_retry_preserves_session_and_presets_can_fail(self):
        self.intercept()
        self.page.goto(API+'/explore')
        self.generate()
        self.nav('Notebook')
        expect(self.page.locator('.history-entry')).to_have_count(1)
        self.responses['/options'] = (503, {'detail':'Constructed options failure'})
        self.responses['/profiles'] = (503, {'detail':'No profiles'})
        self.nav('Explore')
        expect(self.page.get_by_role('alert')).to_contain_text('Constructed options failure')
        self.responses['/options'] = (200, self.fixtures['options'])
        self.page.get_by_role('button', name='Try again').click()
        expect(self.page.get_by_role('combobox')).to_have_count(8)
        expect(self.page.locator('.builder-card .notice')).to_contain_text('select all eight')
        self.nav('Notebook')
        expect(self.page.locator('.history-entry')).to_have_count(1)

    def test_mobile_navigation_keyboard_and_responsive_routes(self):
        self.page.set_viewport_size({'width':390, 'height':844})
        self.page.emulate_media(reduced_motion='reduce')
        self.page.goto(DEMO)
        self.page.keyboard.press('Tab')
        expect(self.page.get_by_role('link', name='Skip to content')).to_be_focused()
        self.page.keyboard.press('Enter')
        expect(self.page.locator('main')).to_be_focused()
        toggle = self.page.get_by_role('button', name='Open navigation')
        toggle.click()
        expect(self.page.get_by_role('button', name='Close navigation')).to_have_attribute('aria-expanded', 'true')
        self.page.keyboard.press('Escape')
        expect(toggle).to_be_focused()
        expect(toggle).to_have_attribute('aria-expanded', 'false')
        self.nav('Explore')
        self.generate()
        self.assert_no_overflow()
        self.nav('Compare')
        self.page.get_by_role('button', name='Compare scenarios', exact=True).click()
        expect(self.page.get_by_role('heading', name='Two contexts, side by side.')).to_be_visible()
        self.assert_no_overflow()
        for width in [320, 390, 768, 1024, 1440]:
            self.page.set_viewport_size({'width':width, 'height':900})
            for route in ['/', '/explore', '/compare', '/scenarios', '/validation', '/how-it-works']:
                self.page.goto(DEMO+route)
                self.page.locator('main h1').wait_for()
                self.assert_no_overflow()

    def test_pending_request_disables_edits_without_fake_progress(self):
        self.intercept()
        pending = []
        self.page.route('http://localhost:8000/generate', lambda route: pending.append(route))
        self.page.goto(API+'/explore')
        self.preset()
        self.page.get_by_role('button', name='Generate baseline', exact=True).click()
        expect(self.page.get_by_role('button', name='Generating…', exact=True)).to_be_disabled()
        expect(self.page.get_by_role('combobox', name='Residence', exact=True)).to_be_disabled()
        expect(self.page.get_by_role('button', name='Urban younger adult')).to_be_disabled()
        expect(self.page.get_by_role('status')).to_contain_text('Results appear when the request completes.')
        expect(self.page.get_by_role('progressbar')).to_have_count(0)
        self.assertEqual(len(pending), 1)
        pending[0].fulfill(status=200, content_type='application/json',
                           body=json.dumps(self.fixtures['generated']),
                           headers={'Access-Control-Allow-Origin':'*'})
        expect(self.page.get_by_role('heading', name='Synthetic Cohort Generated')).to_be_visible()
        expect(self.page.get_by_role('button', name='Generate baseline', exact=True)).to_be_enabled()

    def test_unavailable_health_and_not_found(self):
        self.intercept()
        self.responses['/health'] = (200, {**self.fixtures['health'], 'status':'unavailable', 'mode':'unavailable'})
        self.page.goto(API)
        expect(self.page.locator('.hero-status')).to_contain_text('Model unavailable for generation.')
        self.page.goto(DEMO+'/missing-page')
        expect(self.page.get_by_role('heading', name='This page isn’t in the notebook')).to_be_visible()


if __name__ == '__main__':
    unittest.main(verbosity=2)
