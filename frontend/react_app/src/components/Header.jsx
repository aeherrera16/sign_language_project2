import React from 'react'

export default function Header({ isAuthenticated, onLoginClick, onLogoutClick }){
  return (
    <header className="main-header">
      <div className="header-top">
        <div className="container">
          <div className="logo-section">
            <span className="logo-icon">🤟</span>
            <div className="logo-text">
              <h1>TRADUCTOR LSE</h1>
              <p>Lengua de Señas Ecuatoriana a Tiempo Real</p>
            </div>
          </div>
          
          <div className="header-actions">
            {isAuthenticated ? (
              <button className="btn-login" onClick={onLogoutClick}>
                <span>👤</span> CERRAR SESIÓN
              </button>
            ) : (
              <button className="btn-login" onClick={onLoginClick}>
                <span>👤</span> ADMIN
              </button>
            )}
            <button className="btn-login">
              <span>💬</span> TRADUCIR
            </button>
            <button className="btn-login">
              <span>🕐</span> VER SEÑAS
            </button>
            <button className="btn-login">
              <span>🔍</span>
            </button>
          </div>
        </div>
      </div>
      
      <nav className="main-nav">
        <div className="container">
          <ul className="nav-menu">
            <li><a href="#ayuda" className="active">Ayuda</a></li>
            <li><a href="#senias">Señas</a></li>
            <li><a href="#eventos">Eventos & Noticias</a></li>
            <li><a href="#info">Info</a></li>
          </ul>
        </div>
      </nav>
    </header>
  )
}
