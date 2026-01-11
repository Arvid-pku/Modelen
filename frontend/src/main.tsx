import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from './contexts/ThemeContext'
import App from './App'
import StaticApp from './StaticApp'
import './index.css'

const isStaticMode = import.meta.env.STATIC_MODE === true

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        {isStaticMode ? <StaticApp /> : <App />}
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
