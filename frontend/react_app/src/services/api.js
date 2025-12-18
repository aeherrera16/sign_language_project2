import axios from 'axios'

// Vite exposes environment variables on import.meta.env
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 10000,
})

export default api
