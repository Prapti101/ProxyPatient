"""Run axe WCAG A/AA checks against the frontend demo; see FRONTEND_REVIEW.md."""
from playwright.sync_api import sync_playwright
import json
import os
from pathlib import Path
out=Path(os.environ.get('PP_FRONTEND_ARTIFACTS', '/tmp/proxypatient-frontend-checks'))
out.mkdir(parents=True, exist_ok=True)
base=os.environ.get('PP_FRONTEND_DEMO_URL', 'http://127.0.0.1:4173')
axe_script=os.environ.get('AXE_SCRIPT', '/tmp/proxypatient-a11y/node_modules/axe-core/axe.min.js')
if not Path(axe_script).is_file():
 raise SystemExit('Set AXE_SCRIPT to an installed axe-core/axe.min.js file.')
findings=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH', '/usr/bin/chromium'),args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000})
 def audit(label):
  page.add_script_tag(path=axe_script)
  result=page.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}})")
  violations=[{'id':v['id'],'impact':v['impact'],'description':v['description'],'nodes':[n['target'] for n in v['nodes']]} for v in result['violations']]
  findings.append({'page':label,'violations':violations})
 for route in ['/','/explore','/compare','/scenarios','/validation','/how-it-works']:
  page.goto(base+route,wait_until='networkidle')
  audit(route)
 page.goto(base+'/explore',wait_until='networkidle')
 page.get_by_role('button',name='Urban younger adult').click()
 page.get_by_role('button',name='Generate baseline',exact=True).click()
 page.get_by_role('heading',name='Synthetic Cohort Generated').wait_for()
 audit('generated results')
 page.get_by_role('link',name='Compare',exact=True).click()
 page.get_by_role('button',name='Compare scenarios',exact=True).click()
 page.get_by_role('heading',name='Two contexts, side by side.').wait_for()
 audit('comparison results')
 page.set_viewport_size({'width':390,'height':844})
 page.goto(base,wait_until='networkidle')
 page.get_by_role('button',name='Open navigation').click()
 audit('mobile navigation')
 browser.close()
(out/'accessibility.json').write_text(json.dumps(findings,indent=2))
violations=sum(len(result['violations']) for result in findings)
print(f'{len(findings)} pages/states checked; {violations} WCAG A/AA violations. Details: {out / "accessibility.json"}')
if violations:
 raise SystemExit(1)
