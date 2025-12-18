import React, { useState } from 'react'
import axios from 'axios'

export default function EntrenarModelo(){
  const [entrenando, setEntrenando] = useState(false)
  const [resultado, setResultado] = useState('')
  const [progreso, setProgreso] = useState(0)

  const handleEntrenar = async () => {
    setEntrenando(true)
    setResultado('⏳ Entrenando modelo... Esto puede tomar varios minutos.')
    setProgreso(0)
    
    // Simular progreso
    const interval = setInterval(() => {
      setProgreso(prev => prev < 95 ? prev + 5 : prev)
    }, 1000)
    
    try {
      const res = await axios.post('/api/train')
      clearInterval(interval)
      setProgreso(100)
      setResultado(res.data?.mensaje || '✅ Entrenamiento completado exitosamente.')
    } catch (err) {
      clearInterval(interval)
      setResultado('❌ Error al entrenar el modelo.')
    }
    setEntrenando(false)
  }

  return (
    <div className="admin-content">
      <div className="container">
        <h2>🧠 Entrenar Modelo de IA</h2>
        <p className="section-subtitle">Entrena el modelo de reconocimiento con las señas capturadas</p>
        
        <div className="form-container">
          <div className="info-box">
            <h4>Proceso de Entrenamiento</h4>
            <p>
              El entrenamiento del modelo procesará todas las señas capturadas y creará 
              una red neuronal capaz de reconocer los gestos en tiempo real. Este proceso 
              puede tomar varios minutos dependiendo de la cantidad de datos disponibles.
            </p>
          </div>
          
          <button 
            onClick={handleEntrenar} 
            disabled={entrenando}
            className={entrenando ? 'btn-primary large disabled' : 'btn-primary large'}
          >
            {entrenando ? '⏳ Entrenando...' : '🚀 Iniciar Entrenamiento'}
          </button>
          
          {entrenando && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{width: `${progreso}%`}}></div>
              </div>
              <p className="progress-text">{progreso}% completado</p>
            </div>
          )}
          
          {resultado && (
            <div className={`alert ${resultado.includes('✅') ? 'success' : resultado.includes('❌') ? 'error' : 'warning'}`}>
              {resultado}
            </div>
          )}
          
          <div className="training-info">
            <h4>📊 Información del Entrenamiento</h4>
            <div className="info-grid">
              <div className="info-item">
                <strong>Señas disponibles:</strong>
                <span>12 gestos</span>
              </div>
              <div className="info-item">
                <strong>Muestras totales:</strong>
                <span>360 imágenes</span>
              </div>
              <div className="info-item">
                <strong>Última actualización:</strong>
                <span>Hace 2 días</span>
              </div>
              <div className="info-item">
                <strong>Estado del modelo:</strong>
                <span>Listo para entrenar</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
