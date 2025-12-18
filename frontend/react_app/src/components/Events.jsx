import React from 'react'

export default function Events(){
  return (
    <section className="events-section">
      <div className="container">
        <div className="section-header">
          <h2>Características del Sistema</h2>
          <p>Herramientas avanzadas para traducción de LSE en tiempo real</p>
        </div>
        
        <div className="collections-grid">
          <div className="collection-card yellow-card">
            <div className="card-icon">⚡</div>
            <span className="card-badge">CARACTERÍSTICA</span>
            <div className="card-content">
              <h3>Reconocimiento en Tiempo Real</h3>
              <p>Traducción instantánea de señas capturadas por cámara con alta precisión.</p>
            </div>
          </div>
          
          <div className="collection-card yellow-card">
            <div className="card-icon">�</div>
            <span className="card-badge">CARACTERÍSTICA</span>
            <div className="card-content">
              <h3>Base de Datos Expandible</h3>
              <p>Agrega continuamente nuevas señas al catálogo del sistema.</p>
            </div>
          </div>
          
          <div className="collection-card yellow-card">
            <div className="card-icon">🎯</div>
            <span className="card-badge">CARACTERÍSTICA</span>
            <div className="card-content">
              <h3>Entrenamiento Continuo</h3>
              <p>El modelo de IA mejora su precisión con cada nueva seña registrada.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
