import React, { useState } from 'react'
import axios from 'axios'

export default function AnalizarSenia(){
  const [analizando, setAnalizando] = useState(false)
  const [resultado, setResultado] = useState(null)

  const handleAnalizar = async () => {
    setAnalizando(true)
    setResultado(null)
    try {
      const res = await axios.post('/api/recognize')
      setResultado(res.data)
    } catch (err) {
      setResultado({ error: 'Error al analizar la seña' })
    }
    setAnalizando(false)
  }

  return (
    <div className="admin-content">
      <div className="container">
        <h2>🔬 Analizar y Probar Reconocimiento</h2>
        <p className="section-subtitle">Prueba el modelo de reconocimiento de señas en tiempo real</p>
        
        <div className="form-container">
          <div className="info-box">
            <h4>Prueba de Reconocimiento</h4>
            <p>
              Inicia una sesión de prueba para verificar la precisión del modelo entrenado. 
              El sistema capturará gestos desde tu cámara y mostrará los resultados del 
              análisis en tiempo real con el porcentaje de confianza.
            </p>
          </div>
          
          <div className="camera-preview">
            <div className="camera-placeholder">
              {analizando ? (
                <div className="analyzing-animation">
                  <div className="spinner"></div>
                  <p>Analizando gestos...</p>
                </div>
              ) : (
                <div>
                  <p>🎥</p>
                  <p>Cámara lista</p>
                </div>
              )}
            </div>
          </div>
          
          <button 
            onClick={handleAnalizar} 
            disabled={analizando}
            className={analizando ? 'btn-primary large disabled' : 'btn-primary large'}
          >
            {analizando ? '⏳ Analizando...' : '🎥 Iniciar Prueba'}
          </button>
          
          {resultado && (
            <div className={`result-box ${resultado.error ? 'error' : 'success'}`}>
              {resultado.error ? (
                <p>{resultado.error}</p>
              ) : (
                <div className="result-content">
                  <h4>✅ Resultado del Análisis</h4>
                  <div className="result-details">
                    <div className="result-item">
                      <strong>Gesto detectado:</strong>
                      <span className="detected-gesture">{resultado.gesto || 'No detectado'}</span>
                    </div>
                    <div className="result-item">
                      <strong>Nivel de confianza:</strong>
                      <span className="confidence-level">{resultado.confianza || 0}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{width: `${resultado.confianza || 0}%`}}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
