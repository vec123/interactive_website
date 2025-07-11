import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',   // <= this is the important line to make relative asset paths
  plugins: [react()],
  server: {
    proxy: {
      // Any request starting with /generate_skeleton will be proxied to FastAPI
      '/generate_skeleton': 'http://localhost:8000',
    }
  }
})
