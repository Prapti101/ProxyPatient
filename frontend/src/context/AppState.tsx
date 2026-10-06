import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { Condition, GenerateResponse, ScenarioResult } from '../types/api'
type State = { baseline: Condition|null; setBaseline:(c:Condition)=>void; baselineResult:GenerateResponse|null; setBaselineResult:(r:GenerateResponse)=>void; comparison:ScenarioResult[]; setComparison:(r:ScenarioResult[])=>void; history:GenerateResponse[]; addHistory:(r:GenerateResponse)=>void }
const Ctx = createContext<State|null>(null)
export function AppStateProvider({children}:{children:ReactNode}) {
  const [baseline,setBaselineState]=useState<Condition|null>(null),[baselineResult,setBaselineResult]=useState<GenerateResponse|null>(null),[comparison,setComparison]=useState<ScenarioResult[]>([]),[history,setHistory]=useState<GenerateResponse[]>([])
  const value=useMemo<State>(()=>({baseline,setBaseline:(c)=>setBaselineState(c),baselineResult,setBaselineResult,comparison,setComparison,history,addHistory:(r)=>setHistory((h)=>[...h,r])}),[baseline,baselineResult,comparison,history])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
export function useAppState(){const value=useContext(Ctx);if(!value)throw new Error('AppStateProvider missing');return value}
