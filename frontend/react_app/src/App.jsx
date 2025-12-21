import React, { useState } from 'react'
import Header from './components/Header'
import Hero from './components/Hero'
import Collections from './components/Collections'
import Events from './components/Events'
import Footer from './components/Footer'
import AdminPanel from './components/AdminPanel'
import LoginModal from './components/LoginModal'
import TraductorIA from './components/TraductorIA'
import GrabarSenia from './components/GrabarSenia'
import VerSenas from './components/VerSenas'
import EntrenarModelo from './components/EntrenarModelo'

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [adminView, setAdminView] = useState('dashboard')

  const handleLogin = (username, password) => {
    if (username === 'admin' && password === 'admin123') {
      setIsAuthenticated(true)
      setShowLoginModal(false)
    } else {
      alert('Credenciales incorrectas')
    }
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setAdminView('dashboard')
  }

  const renderContent = () => {
    if (!isAuthenticated) {
      return (
        <>
          <Hero />
          <Collections isAuthenticated={false} adminView={null} />
          <Events />
        </>
      )
    }

    switch (adminView) {
      case 'traductor':
        return <TraductorIA />
      case 'grabar':
        return <GrabarSenia />
      case 'ver':
        return <VerSenas />
      case 'entrenar':
        return <EntrenarModelo />
      case 'dashboard':
      default:
        // Por ahora mostramos Home si no hay componente específico
        // O idealmente un Dashboard real
        return (
          <div style={{ padding: '40px', textAlign: 'center' }}>
            <h2>👋 Bienvenido al Panel de Administración</h2>
            <p>Selecciona una opción del menú lateral.</p>
            <div style={{ marginTop: '20px' }}>
              <p>💡 Usa <b>Traductor IA</b> para probar el reconocimiento.</p>
              <p>💡 Usa <b>Grabar Seña</b> para crear nuevas señas.</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="app">
      <Header
        isAuthenticated={isAuthenticated}
        onLoginClick={() => setShowLoginModal(true)}
        onLogoutClick={handleLogout}
      />

      {showLoginModal && (
        <LoginModal
          onLogin={handleLogin}
          onClose={() => setShowLoginModal(false)}
        />
      )}

      <main className={`main-container ${isAuthenticated ? 'with-admin' : ''}`}>
        {isAuthenticated && (
          <AdminPanel
            activeView={adminView}
            onViewChange={setAdminView}
          />
        )}

        <div className="content-area">
          {renderContent()}
        </div>
      </main>

      <Footer />
    </div>
  )
}
