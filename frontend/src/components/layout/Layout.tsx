import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { ArrowUpRight, Menu, X } from 'lucide-react'
import { dataMode } from '../../services/api'

const links = [
  ['/', 'Overview'],
  ['/explore', 'Explore'],
  ['/compare', 'Compare'],
  ['/scenarios', 'Notebook'],
  ['/validation', 'Validation'],
  ['/how-it-works', 'Method'],
]

export function Layout() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const main = useRef<HTMLElement>(null)
  const menuButton = useRef<HTMLButtonElement>(null)
  const previousPath = useRef(pathname)

  useEffect(() => {
    setOpen(false)
    if (previousPath.current !== pathname) {
      window.scrollTo(0, 0)
      main.current?.focus({ preventScroll: true })
      previousPath.current = pathname
    }
    document.title = `${links.find(([path]) => path === pathname)?.[1] ?? 'Overview'} · ProxyPatient`
  }, [pathname])

  useEffect(() => {
    function close(event: KeyboardEvent) {
      if (event.key === 'Escape' && open) {
        setOpen(false)
        menuButton.current?.focus()
      }
    }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [open])

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink
            to="/"
            className="brand"
            onClick={() => setOpen(false)}
            aria-label="ProxyPatient overview"
          >
            <span className="brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            <span>
              <b>
                ProxyPatient<span className="brand-period">.</span>
              </b>
              <small>Population research notebook</small>
            </span>
          </NavLink>
          <button
            ref={menuButton}
            className="mobile-menu"
            aria-label={open ? 'Close navigation' : 'Open navigation'}
            aria-expanded={open}
            aria-controls="primary-navigation"
            onClick={() => setOpen(!open)}
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
          <nav
            id="primary-navigation"
            aria-label="Main navigation"
            className={open ? 'nav open' : 'nav'}
          >
            {links.map(([to, label]) => (
              <NavLink end={to === '/'} to={to} key={to} onClick={() => setOpen(false)}>
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="session-strip">
          <div>
            <span>
              <span className="mode-dot" />
              {dataMode === 'demo' ? 'Interface demo' : 'API mode'}
            </span>
            <span>
              {dataMode === 'demo'
                ? 'Illustrative values · no fitted survey model'
                : 'Check the run label alongside each result'}
            </span>
            <NavLink to="/how-it-works">
              Read the method <ArrowUpRight size={12} />
            </NavLink>
          </div>
        </div>
      </header>
      <main id="main-content" ref={main} tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="footer">
        <div>
          <b>ProxyPatient.</b>
          <span>Synthetic scenarios. Considered interpretation.</span>
        </div>
        <p>
          For research and education.
          <br />
          Group summaries are not personal medical predictions.
        </p>
      </footer>
    </div>
  )
}
