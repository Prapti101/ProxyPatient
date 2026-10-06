import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { AppStateProvider } from './context/AppState'
import { Layout } from './components/layout/Layout'
import { Empty, ExploreLink, Loading } from './components/ui/Blocks'
const Home = lazy(() => import('./pages/Home'))
const Explore = lazy(() => import('./pages/Explore'))
const Scenarios = lazy(() => import('./pages/Scenarios'))
const Compare = lazy(() => import('./pages/Compare'))
const Validation = lazy(() => import('./pages/Validation'))
const HowItWorks = lazy(() => import('./pages/HowItWorks'))
export default function App() {
  return (
    <BrowserRouter>
      <AppStateProvider>
        <Routes>
          <Route element={<Layout />}>
            {[
              { path: '/', component: <Home /> },
              { path: '/explore', component: <Explore /> },
              { path: '/scenarios', component: <Scenarios /> },
              { path: '/compare', component: <Compare /> },
              { path: '/validation', component: <Validation /> },
              { path: '/how-it-works', component: <HowItWorks /> },
            ].map(({ path, component }) => (
              <Route
                key={path}
                path={path}
                element={
                  <Suspense
                    fallback={
                      <div className="page">
                        <Loading label="Opening page…" />
                      </div>
                    }
                  >
                    {component}
                  </Suspense>
                }
              />
            ))}
            <Route
              path="*"
              element={
                <div className="page">
                  <Empty title="This page isn’t in the notebook" action={<ExploreLink />}>
                    Use the navigation to return to your workspace.
                  </Empty>
                </div>
              }
            />
          </Route>
        </Routes>
      </AppStateProvider>
    </BrowserRouter>
  )
}
