import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { Condition, GenerateResponse, ScenarioResult } from '../types/api'

type State = {
  baseline: Condition | null
  baselineResult: GenerateResponse | null
  acceptBaseline: (condition: Condition, result: GenerateResponse) => void
  comparison: ScenarioResult[]
  setComparison: (results: ScenarioResult[]) => void
  history: GenerateResponse[]
  addHistory: (result: GenerateResponse) => void
}
const Context = createContext<State | null>(null)
export function AppStateProvider({ children }: { children: ReactNode }) {
  const [reference, setReference] = useState<{
    condition: Condition
    result: GenerateResponse
  } | null>(null)
  const [comparison, setComparison] = useState<ScenarioResult[]>([])
  const [history, setHistory] = useState<GenerateResponse[]>([])
  const value = useMemo<State>(
    () => ({
      baseline: reference?.condition ?? null,
      baselineResult: reference?.result ?? null,
      acceptBaseline: (condition, result) => {
        setReference({ condition: { ...condition }, result })
        setComparison([])
        setHistory((previous) => [...previous, result])
      },
      comparison,
      setComparison,
      history,
      addHistory: (result) => setHistory((previous) => [...previous, result]),
    }),
    [reference, comparison, history],
  )
  return <Context.Provider value={value}>{children}</Context.Provider>
}
export function useAppState() {
  const value = useContext(Context)
  if (!value) throw new Error('AppStateProvider missing')
  return value
}
