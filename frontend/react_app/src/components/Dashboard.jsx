import React, { useEffect, useState } from 'react'
import '../styles/admin.css'

export default function Dashboard(){
  const [stats, setStats] = useState({
    gestures: 12,
    model: true,
    system: 'Operativo',
    accuracy: 94.5
  })

  return (
    <div className="admin-wrap">
      <aside className="admin-sidebar">
        {/* Sidebar placeholder - mantiene navegación existente */}
        <div className="brand">Traductor LSE</div>
        <nav className="nav-list">
          <a className="nav-item active">Dashboard</a>
          <a className="nav-item">Captura IA</a>
          <a className="nav-item">Entrenar Modelo</a>
          <a className="nav-item">Ver Señas</a>
          <a className="nav-item">Ajustes</a>
        </nav>
        <div className="sidebar-footer">Admin</div>
      </aside>

      <main className="admin-main">
        <header className="admin-hero">
          <div className="hero-left">
            <h1>Panel de Administración</h1>
            <p className="hero-sub">Monitorea el sistema de reconocimiento LSE y captura señas con IA integrada</p>
            <div className="hero-actions">
              <button className="btn primary">Abrir Captura IA</button>
              <button className="btn outline">Entrenar Modelo</button>
            </div>
          </div>
          <div className="hero-right">
            <div className="status-chip">Estado: <strong>{stats.system}</strong></div>
            <div className="accuracy-chip">Precisión: <strong>{stats.accuracy}%</strong></div>
          </div>
        </header>

        <section className="stats-grid">
          <div className="stat">
            <div className="stat-number">{stats.gestures}</div>
            <div className="stat-desc">Señas capturadas</div>
          </div>
          <div className="stat">
            <div className="stat-number">{stats.model ? '✓' : '○'}</div>
            <div className="stat-desc">Modelo entrenado</div>
          </div>
          <div className="stat">
            <div className="stat-number">{stats.system}</div>
            <div className="stat-desc">Estado sistema</div>
          </div>
          <div className="stat">
            <div className="stat-number">{stats.accuracy}%</div>
            <div className="stat-desc">Precisión</div>
          </div>
        </section>

        <section className="cards-row">
          <div className="card big-card">
            <h3>Acciones Rápidas</h3>
            <p>Usa los botones para capturar, analizar y entrenar sin salir del panel.</p>
            <ul>
              <li>📹 <strong>Grabar Seña</strong> - Captura nuevas muestras</li>
              <li>🧠 <strong>Entrenar</strong> - Entrena el modelo con datos válidos</li>
              <li>🤖 <strong>Analizar IA</strong> - Feedback de calidad instantáneo</li>
            </ul>
          </div>

          <div className="card">
            <h4>Actividad Reciente</h4>
            <ul className="recent-list">
              <li>Captura: "HOLA" - Puntaje 82</li>
              <li>Entrenamiento: 12 nuevas muestras</li>
              <li>Backup automático: 02/12/2025</li>
            </ul>
          </div>

          <div className="card">
            <h4>Notificaciones</h4>
            <p>No hay alertas críticas</p>
          </div>

          <div className="card">
            <h4>Diccionario de referencia</h4>
            <p>Usa el diccionario oficial para mapear y etiquetar las señas principales del proyecto.</p>
            <a className="btn outline" href="http://www.plataformaconadis.gob.ec/~platafor/diccionario/" target="_blank" rel="noopener noreferrer">Abrir Diccionario CONADIS</a>
            <p className="muted" style={{marginTop:8,fontSize:12}}>Fuente: plataforma CONADIS - Ecuador</p>
          </div>
        </section>

      </main>
    </div>
  )
}
