import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Preserve workspace paths: the managed Windows runner restricts native realpath
// calls even for files inside the granted workspace.
export default defineConfig({ plugins: [react()], resolve: { preserveSymlinks: true } })
