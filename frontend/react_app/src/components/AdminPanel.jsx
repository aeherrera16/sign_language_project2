import React from 'react'

export default function AdminPanel({ activeView, onViewChange }) {
  const menuItems = [
    { id: 'dashboard', icon: '📊', label: 'Dashboard' },
    { id: 'traductor', icon: '🧠', label: 'Traductor IA' },
    { id: 'grabar', icon: '📹', label: 'Grabar Seña' },
    { id: 'entrenar', icon: '⚙️', label: 'Entrenar Modelo' },
    { id: 'analizar', icon: '🔬', label: 'Analizar/Pruebas' },
    { id: 'ver', icon: '👁️', label: 'Ver Señas' }
  ]

  return (
    <aside className="admin-panel">
      <div className="admin-panel-header">
        <h3>🔧 Panel de Administración</h3>
      </div>

      <nav className="admin-menu">
        <ul>
          {menuItems.map(item => (
            <li key={item.id}>
              <button
                className={activeView === item.id ? 'active' : ''}
                onClick={() => onViewChange(item.id)}
              >
                <span className="menu-icon">{item.icon}</span>
                <span className="menu-label">{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="admin-footer">
        <p>Sesión activa</p>
        <p className="admin-user">👤 Admin</p>
      </div>
    </aside>
  )
}
