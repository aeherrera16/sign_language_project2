import React, { useState, useEffect } from 'react'
import api from '../services/api'
import Modal from './Modal'

export default function EntrenarModelo() {
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)
  const [history, setHistory] = useState(null)
  const [progress, setProgress] = useState(0)
  const [modal, setModal] = useState({ open: false, title: '', message: '', type: 'info' })
  const [availableGestures, setAvailableGestures] = useState([])

  // Función helper para mostrar modal
  const showModal = (title, message, type = 'info') => {
    setModal({ open: true, title, message, type })
  }

  // Polling para estado del entrenamiento
  useEffect(() => {
    let interval
    if (loading) {
      interval = setInterval(checkStatus, 1000)
    }
    return () => clearInterval(interval)
  }, [loading])

  // Cargar historial y señas disponibles al inicio
  useEffect(() => {
    fetchHistory()
    fetchGestures()
  }, [])

  const fetchGestures = async () => {
    try {
      const res = await api.get('/gestures/list')
      setAvailableGestures(res.data.gestures || [])
    } catch (err) {
      console.error("Error cargando gestos disponibles", err)
    }
  }

  const fetchHistory = async () => {
    try {
      const res = await api.get(`/training/history?t=${Date.now()}`)
      setHistory(res.data)
    } catch (e) {
      console.log("No hay historial previo")
    }
  }

  const checkStatus = async () => {
    try {
      const res = await api.get('/training/status')
      const data = res.data

      setStatus(data.message)
      setProgress(data.progress)

      if (!data.is_training && loading) {
        setLoading(false)
        fetchHistory()
        if (data.progress === 100) {
          showModal('¡Éxito!', 'Entrenamiento completado exitosamente. Ahora puedes usar el Traductor IA.', 'success')
        }
      }
    } catch (err) {
      console.error(err)
    }
  }

  const startTraining = async () => {
    if (availableGestures.length < 2) {
      showModal('Faltan datos', 'Necesitas grabar al menos 2 señas diferentes para entrenar un modelo útil.', 'warning')
      return
    }

    try {
      setLoading(true)
      setProgress(0)
      setStatus('Iniciando...')

      await api.post('/training/start')
    } catch (err) {
      setLoading(false)
      const msg = err.response?.data?.detail || 'Error al iniciar entrenamiento'
      showModal('Error', msg, 'error')
    }
  }

  return (
    <div className="admin-content">
      {modal.open && (
        <Modal
          title={modal.title}
          message={modal.message}
          type={modal.type}
          onClose={() => setModal({ ...modal, open: false })}
        />
      )}

      <div className="container">
        <h2>🧠 Entrenar Modelo IA</h2>
        <p>Crea el "cerebro" artificial usando las señas que has grabado.</p>

        <div className="training-panel" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginTop: '30px' }}>

          {/* Panel Izquierdo: Control */}
          <div className="control-card" style={{ background: 'white', padding: '30px', borderRadius: '15px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)' }}>
            <h3>⚙️ Panel de Control</h3>

            <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '10px', marginBottom: '20px', border: '1px solid #e9ecef' }}>
              <h4 style={{ marginTop: 0, marginBottom: '10px', color: '#495057' }}>📂 Datos detectados en el servidor:</h4>
              {availableGestures.length === 0 ? (
                <p style={{ color: '#666', fontStyle: 'italic', margin: 0 }}>No hay señas para entrenar.</p>
              ) : (
                <ul style={{ margin: 0, paddingLeft: '20px', maxHeight: '150px', overflowY: 'auto' }}>
                  {availableGestures.map(g => (
                    <li key={g.name} style={{ marginBottom: '5px', color: '#212529' }}>
                      <b>{g.name}</b> <span style={{ color: '#6c757d', fontSize: '0.9em' }}>({g.samples} img)</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <p style={{ color: '#666', marginBottom: '20px', fontSize: '0.9em' }}>
              Se generará un modelo nuevo usando estos datos.
            </p>

            {loading ? (
              <div className="progress-section">
                <div className="progress-bar-bg" style={{ width: '100%', height: '20px', background: '#e9ecef', borderRadius: '10px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${progress}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                      transition: 'width 0.5s ease'
                    }}
                  />
                </div>
                <p style={{ textAlign: 'center', marginTop: '10px', fontWeight: 'bold', color: '#3b82f6' }}>
                  {status} ({progress}%)
                </p>
                <p style={{ fontSize: '0.9em', color: '#888', textAlign: 'center' }}>
                  Por favor espera...
                </p>
              </div>
            ) : (
              <button
                onClick={startTraining}
                className="btn-primary"
                style={{
                  width: '100%',
                  padding: '15px',
                  fontSize: '1.2em',
                  background: 'linear-gradient(135deg, #0d6efd 0%, #0043a8 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  boxShadow: '0 4px 10px rgba(13, 110, 253, 0.3)',
                  opacity: availableGestures.length < 2 ? 0.6 : 1,
                  pointerEvents: availableGestures.length < 2 ? 'none' : 'auto'
                }}
              >
                {availableGestures.length < 2 ? '⚠️ Se necesitan 2+ señas' : '🚀 Iniciar Entrenamiento'}
              </button>
            )}
          </div>

          {/* Panel Derecho: Estadísticas */}
          <div className="stats-card" style={{ background: '#f8f9fa', padding: '30px', borderRadius: '15px', border: '1px solid #e9ecef' }}>
            <h3>📊 Estado del Modelo Actual</h3>

            {history ? (
              <div className="metrics">
                <div className="metric-item" style={{ marginBottom: '15px' }}>
                  <label>Precisión (Accuracy):</label>
                  <div style={{ fontSize: '2em', color: '#198754', fontWeight: 'bold' }}>
                    {((history.final_accuracy ?? (history.accuracy?.[history.accuracy.length - 1] ?? 0)) * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="metric-item" style={{ marginBottom: '15px' }}>
                  <label>Pérdida (Loss):</label>
                  <div style={{ fontSize: '1.2em', color: '#666' }}>
                    {(history.final_loss ?? (history.loss?.[history.loss.length - 1] ?? 0)).toFixed(4)}
                  </div>
                </div>

                <div className="metric-item">
                  <label>Último entrenamiento:</label>
                  <div style={{ fontSize: '0.9em', color: '#666' }}>
                    {history.timestamp ? new Date(history.timestamp).toLocaleString() : 'Desconocido'}
                  </div>
                </div>

                <hr style={{ margin: '20px 0', borderColor: '#dfe2e5' }} />

                <h4>Señas Aprendidas ({history.num_gestures || history.gestures?.length || 0}):</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                  {(history.gestures || []).map(g => (
                    <span key={g} style={{ background: '#e2e8f0', padding: '4px 8px', borderRadius: '4px', fontSize: '0.9em' }}>
                      {g}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#888', marginTop: '40px' }}>
                <p>No se ha entrenado ningún modelo aún.</p>
                <p>Las estadísticas aparecerán aquí después de entrenar.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
