import React, { useEffect, useState } from 'react'
import axios from 'axios'

export default function VerSenias(){
  const [senias, setSenias] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchSenias() {
      try {
        const res = await axios.get('/api/gestures')
        setSenias(res.data || [])
      } catch (err) {
        // Datos de demostración si no hay backend
        setSenias([
          { nombre: 'Hola', samples: 30, fecha: '2025-11-28' },
          { nombre: 'Gracias', samples: 28, fecha: '2025-11-27' },
          { nombre: 'Por favor', samples: 32, fecha: '2025-11-26' },
          { nombre: 'Adiós', samples: 25, fecha: '2025-11-25' },
          { nombre: 'Sí', samples: 30, fecha: '2025-11-24' },
          { nombre: 'No', samples: 30, fecha: '2025-11-24' },
        ])
      }
      setLoading(false)
    }
    fetchSenias()
  }, [])

  return (
    <div className="admin-content">
      <div className="container">
        <h2>👁️ Señas Registradas</h2>
        <p className="section-subtitle">Biblioteca completa de señas capturadas en el sistema</p>
        
        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Cargando señas...</p>
          </div>
        ) : senias.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📭</div>
            <h3>No hay señas registradas</h3>
            <p>Comienza grabando tu primera seña desde el menú "Grabar Seña"</p>
          </div>
        ) : (
          <div className="senias-grid">
            {senias.map((senia, index) => (
              <div key={index} className="senia-card">
                <div className="senia-icon">🤟</div>
                <div className="senia-info">
                  <h3>{senia.nombre}</h3>
                  <p className="senia-samples">{senia.samples || 0} muestras</p>
                  <p className="senia-date">Registrado: {senia.fecha || 'Fecha desconocida'}</p>
                </div>
                <div className="senia-actions">
                  <button className="btn-icon" title="Ver detalles">👁️</button>
                  <button className="btn-icon" title="Editar">✏️</button>
                  <button className="btn-icon" title="Eliminar">🗑️</button>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="senias-stats">
          <div className="stat-item">
            <strong>Total de señas:</strong>
            <span>{senias.length}</span>
          </div>
          <div className="stat-item">
            <strong>Total de muestras:</strong>
            <span>{senias.reduce((acc, s) => acc + (s.samples || 0), 0)}</span>
          </div>
          <div className="stat-item">
            <strong>Última actualización:</strong>
            <span>{senias[0]?.fecha || 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
