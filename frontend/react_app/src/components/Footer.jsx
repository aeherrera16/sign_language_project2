import React from 'react'

export default function Footer(){
  return (
    <footer className="main-footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Traductor LSE</h4>
            <p>Traducción de Lengua de Señas Ecuatoriana a Tiempo Real</p>
            <div className="social-links">
              <a href="#" aria-label="Facebook">📘</a>
              <a href="#" aria-label="Twitter">🐦</a>
              <a href="#" aria-label="Instagram">📷</a>
              <a href="#" aria-label="YouTube">📹</a>
            </div>
          </div>
          
          <div className="footer-section">
            <h4>Sistema</h4>
            <ul>
              <li><a href="#senias">Catálogo de Señas</a></li>
              <li><a href="#modelo">Modelo de IA</a></li>
              <li><a href="#precision">Precisión</a></li>
              <li><a href="#estadisticas">Estadísticas</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Ayuda</h4>
            <ul>
              <li><a href="#faq">Preguntas Frecuentes</a></li>
              <li><a href="#soporte">Soporte Técnico</a></li>
              <li><a href="#guias">Guías</a></li>
              <li><a href="#contacto">Contacto</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Contacto</h4>
            <p>📧 info@traductorLSE.ec</p>
            <p>📞 +593 2 123 4567</p>
            <p>📍 Quito, Ecuador</p>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>© 2025 Traductor LSE. Todos los derechos reservados.</p>
          <div className="footer-legal">
            <a href="#privacidad">Política de Privacidad</a>
            <span>|</span>
            <a href="#terminos">Términos de Uso</a>
            <span>|</span>
            <a href="#accesibilidad">Accesibilidad</a>
          </div>
        </div>
      </div>
    </footer>
  )
}
