import React, { useState } from 'react'

export default function LoginModal({ onLogin, onClose }){
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onLogin(username, password)
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <h2>Iniciar Sesión</h2>
        <p className="modal-subtitle">Panel de Administración LSE</p>
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label>Usuario</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Ingresa tu usuario"
              autoFocus
            />
          </div>
          
          <div className="form-group">
            <label>Contraseña</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Ingresa tu contraseña"
            />
          </div>
          
          <button type="submit" className="btn-primary">
            Iniciar Sesión
          </button>
          
          <p className="helper-text">
            Usuario de prueba: <strong>admin</strong> / <strong>admin123</strong>
          </p>
        </form>
      </div>
    </div>
  )
}
