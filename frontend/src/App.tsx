import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { AppStateProvider } from './context/AppState'
import { Layout } from './components/layout/Layout'
import { Loading } from './components/ui/Blocks'
const Home=lazy(()=>import('./pages/Home'))
const Explore=lazy(()=>import('./pages/Explore'))
const Scenarios=lazy(()=>import('./pages/Scenarios'))
const Compare=lazy(()=>import('./pages/Compare'))
const Validation=lazy(()=>import('./pages/Validation'))
const HowItWorks=lazy(()=>import('./pages/HowItWorks'))
export default function App(){return <BrowserRouter><AppStateProvider><Suspense fallback={<div className="page"><Loading label="Opening page…"/></div>}><Routes><Route element={<Layout/>}><Route index element={<Home/>}/><Route path="explore" element={<Explore/>}/><Route path="scenarios" element={<Scenarios/>}/><Route path="compare" element={<Compare/>}/><Route path="validation" element={<Validation/>}/><Route path="how-it-works" element={<HowItWorks/>}/><Route path="*" element={<Home/>}/></Route></Routes></Suspense></AppStateProvider></BrowserRouter>}
