import React, { useState } from 'react'
import Header from './components/Header'
import Hero from './components/Hero'
import Collections from './components/Collections'
import Events from './components/Events'
import Footer from './components/Footer'
import AdminPanel from './components/AdminPanel'
import LoginModal from './components/LoginModal'
import CapturaIA from './components/CapturaIA'

export default function App(){
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
          {isAuthenticated && adminView === 'captura-ia' ? (
            <CapturaIA />
          ) : (
            <>
              <Hero />
              <Collections isAuthenticated={isAuthenticated} adminView={adminView} />
              <Events />
            </>
          )}
        </div>
      </main>

      <Footer />
    </div>
  )
}
