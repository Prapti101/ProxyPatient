import * as demo from '../mocks/demoData'
import { request } from './client'
import type { CompareRequest, CompareResponse, GenerateRequest, GenerateResponse, HealthResponse, OptionsResponse, ParseRequest, ParseResponse, ProfilesResponse, SchemaResponse, ValidationResponse } from '../types/api'
const configuredMode = (import.meta.env.VITE_DATA_MODE || 'demo').toLowerCase()
if (configuredMode !== 'api' && configuredMode !== 'demo') throw new Error('VITE_DATA_MODE must be either "demo" or "api".')
export const dataMode = configuredMode as 'api'|'demo'
let counter = 0
const api = {
  health: () => dataMode === 'demo' ? Promise.resolve(demo.health) : request<HealthResponse>('/health'),
  getSchema: () => dataMode === 'demo' ? Promise.resolve({ supported_state_codes:demo.options.state } as SchemaResponse) : request<SchemaResponse>('/schema'),
  getOptions: () => dataMode === 'demo' ? Promise.resolve(demo.options) : request<OptionsResponse>('/options'),
  getProfiles: () => dataMode === 'demo' ? Promise.resolve(demo.profileResponse) : request<ProfilesResponse>('/profiles'),
  generate: (input: GenerateRequest) => dataMode === 'demo' ? Promise.resolve(demo.demoGenerate(input.condition,`S${++counter}`)) : request<GenerateResponse>('/generate',{method:'POST',body:JSON.stringify(input)}),
  compare: (input: CompareRequest) => dataMode === 'demo' ? Promise.resolve((() => { const generated=input.scenarios.map(s=>demo.demoGenerate(s.condition,`S${++counter}`)); const scenarios=generated.map((r,i)=>({...r,label:input.scenarios[i].label,delta_pp:i===0?0:(r.outcome_stat.rate-generated[0].outcome_stat.rate)*100})); return {scenarios,run_type:'mock' as const,preliminary:false,demo:true,status_banner:'DEMO (mock data)',model_fingerprint:'demo-adapter',disclaimer:'Synthetic scenario comparison; not causal.',weighting:'unweighted sample',uncertainty_note:'Demo adapter only.',note:'Descriptive synthetic scenario comparison; what-if is not causal.'} })()) : request<CompareResponse>('/compare',{method:'POST',body:JSON.stringify(input)}),
  getValidation: () => dataMode === 'demo' ? Promise.resolve(demo.validation) : request<ValidationResponse>('/validation'),
  getModelComparison: () => dataMode === 'demo' ? Promise.resolve(demo.validation) : request<ValidationResponse>('/model-comparison'),
  parse: (input: ParseRequest) => dataMode === 'demo' ? Promise.reject(new Error('Text proposals are available only when the API is explicitly running in demo mode.')) : request<ParseResponse>('/parse',{method:'POST',body:JSON.stringify(input)})
}
export default api
