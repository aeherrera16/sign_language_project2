import React, { useState } from 'react'
import axios from 'axios'

export default function GrabarSenia(){
  const [nombre, setNombre] = useState('')
  const [estado, setEstado] = useState('')
  const [grabando, setGrabando] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!nombre.trim()) {
      setEstado('⚠️ Debes ingresar un nombre para la seña.')
      return
    }
    setGrabando(true)
    setEstado('⏳ Iniciando grabación...')
    try {
      const res = await axios.post('/api/gestures', { nombre })
      setEstado(res.data?.mensaje || '✅ Grabación iniciada correctamente.')
    } catch (err) {
      setEstado('❌ Error al iniciar la grabación.')
    }
    setGrabando(false)
  }

  return (
    <div className="admin-content">
      <div className="container">
        <h2>📹 Grabar Nueva Seña</h2>
        <p className="section-subtitle">Captura un nuevo gesto para entrenar el modelo de reconocimiento</p>
        
        <div className="form-container">
          <form onSubmit={handleSubmit} className="admin-form">
            <div className="form-group">
              <label>Nombre de la seña</label>
              <input
                type="text"
                placeholder="Ejemplo: hola, gracias, por favor..."
                value={nombre}
                onChange={e => setNombre(e.target.value)}
                disabled={grabando}
              />
              <small>Ingresa el nombre descriptivo de la seña que deseas grabar</small>
            </div>
            
            <button 
              type="submit" 
              disabled={grabando}
              className={grabando ? 'btn-primary disabled' : 'btn-primary'}
            >
              {grabando ? '⏳ Grabando...' : '🎥 Iniciar Grabación'}
            </button>
          </form>
          
          {estado && (
            <div className={`alert ${estado.includes('✅') ? 'success' : estado.includes('❌') ? 'error' : 'warning'}`}>
              {estado}
            </div>
          )}
          
          <div className="info-box">
            <h4>💡 Instrucciones</h4>
            <ol>
              <li>Ingresa el nombre de la seña que deseas registrar</li>
              <li>Haz clic en "Iniciar Grabación"</li>
              <li>Realiza la seña frente a la cámara múltiples veces</li>
              <li>El sistema capturará automáticamente las muestras</li>
              <li>Repite con diferentes variaciones de la misma seña</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}
