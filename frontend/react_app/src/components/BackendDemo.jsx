import React, { useState } from 'react'

export default function BackendDemo() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleTest = async () => {
    setLoading(true)
    setResult(null)
    setTimeout(() => {
      setResult('👋 ¡Reconocimiento exitoso! El sistema detectó el gesto de saludo.');
      setLoading(false);
    }, 1200);
  }

  return (
    <section className="container" style={{marginTop:40, maxWidth:600}}>
      <div style={{background:'#fff',borderRadius:12,boxShadow:'0 2px 12px 0 rgba(0,0,0,0.06)',padding:'2rem'}}>
        <h3 style={{color:'var(--primary)',fontWeight:700,marginTop:0}}>Demo interactiva: ¿Cómo funciona?</h3>
        <p style={{color:'var(--muted)'}}>Haz clic en el botón para simular el reconocimiento de un gesto de lengua de señas.</p>
        <button onClick={handleTest} disabled={loading} style={{background:'var(--accent)',color:'var(--primary)',fontWeight:700,padding:'0.7rem 1.5rem',border:'none',borderRadius:6,fontSize:'1.1rem',marginTop:10,cursor:'pointer'}}>
          {loading ? 'Reconociendo...' : 'Probar reconocimiento'}
        </button>
        {result && <div style={{marginTop:18, color:'var(--primary)',fontSize:'1.2rem',fontWeight:600}}>{result}</div>}
      </div>
    </section>
  )
}
