import React, { useState } from 'react'
import axios from 'axios'
import InfoCarousel from './InfoCarousel'
import GrabarSenia from './GrabarSenia'
import EntrenarModelo from './EntrenarModelo'
import AnalizarSenia from './AnalizarSenia'
import VerSenias from './VerSenias'
import Dashboard from './Dashboard'

export default function Collections({ isAuthenticated, adminView }){
  
  if (isAuthenticated) {
    return (
      <section className="collections-section admin-view">
        {adminView === 'dashboard' && <Dashboard />}
        {adminView === 'grabar' && <GrabarSenia />}
        {adminView === 'entrenar' && <EntrenarModelo />}
        {adminView === 'analizar' && <AnalizarSenia />}
        {adminView === 'ver' && <VerSenias />}
      </section>
    )
  }

  return (
    <section className="collections-section">
      <div className="container">
        <div className="section-header">
          <h2>Usando el Traductor</h2>
          <p>Captura y reconoce Lengua de Señas Ecuatoriana en tiempo real</p>
        </div>
        
        <InfoCarousel />
        
        <div className="collections-grid">
          <div className="collection-card">
            <div className="card-icon">🎥</div>
            <span className="card-badge">CÓMO HACER</span>
            <div className="card-content">
              <h3>Capturar nuevas señas</h3>
              <p>Registra y almacena nuevas señas de Lengua de Señas Ecuatoriana.</p>
            </div>
          </div>
          
          <div className="collection-card">
            <div className="card-icon">🤖</div>
            <span className="card-badge">CÓMO HACER</span>
            <div className="card-content">
              <h3>Entrenar el modelo de IA</h3>
              <p>Mejora la precisión del reconocimiento con más datos de entrenamiento.</p>
            </div>
          </div>
          
          <div className="collection-card yellow-card">
            <div className="card-icon">🔍</div>
            <span className="card-badge">CÓMO HACER</span>
            <div className="card-content">
              <h3>Reconocer señas en tiempo real</h3>
              <p>Traduce señas capturadas por la cámara en tiempo real.</p>
            </div>
          </div>
          
          <div className="collection-card">
            <div className="card-icon">📊</div>
            <span className="card-badge">CÓMO HACER</span>
            <div className="card-content">
              <h3>Ver todas las señas registradas</h3>
              <p>Explora el catálogo completo de señas disponibles en el sistema.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
