import React from 'react'

export default function Hero(){
  const today = new Date()
  const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }
  const dateString = today.toLocaleDateString('es-ES', options).toUpperCase()
  
  return (
    <section className="hero">
      <div className="hero-content">
        <h2>
          <strong>TRADUCTOR DE LENGUA DE SEÑAS ECUATORIANA A TIEMPO REAL</strong>
        </h2>
        <p>
          Captura, entrena y reconoce señas de Lengua de Señas Ecuatoriana en tiempo real
        </p>
        <div className="hero-actions">
          <button className="btn-primary">COMENZAR TRADUCCIÓN</button>
        </div>
        <div style={{marginTop: '3rem'}}>
          <span className="hero-date">{dateString}</span>
        </div>
      </div>
    </section>
  )
}
