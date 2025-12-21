import React, { useState, useRef, useEffect, useCallback } from 'react'
import api from '../services/api'

export default function GrabarSenia() {
  const [nombre, setNombre] = useState('')
  const [estado, setEstado] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [samplesCount, setSamplesCount] = useState(0)
  const [lastCapture, setLastCapture] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  // Iniciar cámara al montar
  useEffect(() => {
    startCamera()
    return () => stopCamera()
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setEstado('📸 Cámara activa. Listo para grabar.')
    } catch (err) {
      console.error(err)
      setEstado('❌ Error: No se pudo acceder a la cámara')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
  }

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isRecording) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    // Dibujar frame actual en canvas
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    // Convertir a blob
    canvas.toBlob(async (blob) => {
      if (!blob) return

      const formData = new FormData()
      formData.append('video', blob, 'capture.jpg')
      formData.append('gesture_name', nombre)

      try {
        const res = await api.post('/gestures/capture', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        if (res.data.success) {
          setSamplesCount(prev => prev + 1)
          const imgToShow = res.data.segmented_image || URL.createObjectURL(blob)
          setLastCapture(imgToShow)
          setEstado('✅ ¡Guardado!')
          flashBorder('green')
        } else {
          // Caso: No se detectaron manos
          setEstado('🔎 Buscando manos...')
        }
      } catch (err) {
        console.error('Error:', err)
        setEstado('⚠️ Error de conexión')
      }
    }, 'image/jpeg', 0.9)
  }, [isRecording, nombre])

  const flashBorder = (color) => {
    const video = videoRef.current
    if (video) {
      video.style.border = `4px solid ${color}`
      setTimeout(() => {
        video.style.border = 'none'
      }, 150)
    }
  }

  // Loop de grabación
  useEffect(() => {
    let intervalId
    if (isRecording && nombre.trim()) {
      // No reseteamos estado aquí para no borrar errores
      // Capturar cada 300ms (un poco más lento para procesar bien)
      intervalId = setInterval(captureFrame, 300)
    } else {
      clearInterval(intervalId)
      if (videoRef.current) videoRef.current.style.border = 'none'
    }
    return () => clearInterval(intervalId)
  }, [isRecording, nombre, captureFrame])

  const toggleRecording = (e) => {
    e.preventDefault()
    if (!nombre.trim()) {
      setEstado('⚠️ Escribe un nombre para la seña primero')
      return
    }
    setIsRecording(!isRecording)
    if (!isRecording) setEstado('🎥 Iniciando...')
  }

  return (
    <div className="admin-content">
      <div className="container">
        <h2>📹 Grabar Nueva Seña</h2>

        <div className="recording-layout" style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '20px' }}>

          {/* Columna Izquierda: Cámara */}
          <div className="camera-section">
            <div className="video-container" style={{ position: 'relative', background: '#000', borderRadius: '12px', overflow: 'hidden', minHeight: '480px' }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', transition: 'border 0.1s' }}
              />
              {isRecording && (
                <div style={{ position: 'absolute', top: 10, right: 10, background: 'red', color: 'white', padding: '5px 10px', borderRadius: '20px', animation: 'pulse 1s infinite' }}>
                  🔴 GRABANDO
                </div>
              )}
            </div>
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>

          {/* Columna Derecha: Controles */}
          <div className="controls-section">
            <div className="form-group">
              <label>Nombre de la Seña</label>
              <input
                type="text"
                value={nombre}
                onChange={e => setNombre(e.target.value)}
                placeholder="Ej: hola, gracias..."
                disabled={isRecording}
                style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #ccc', marginBottom: '15px' }}
              />
            </div>

            <div className="stats-box" style={{ background: '#f5f5f5', padding: '15px', borderRadius: '8px', marginBottom: '15px', textAlign: 'center' }}>
              <h4>Muestras: {samplesCount}</h4>
              <p style={{ fontSize: '0.9em', color: '#666' }}>Se recomiendan 30-50 muestras</p>
            </div>

            <div style={{ display: 'flex', gap: '10px', flexDirection: 'column' }}>
              <button
                onClick={toggleRecording}
                className={`btn-block ${isRecording ? 'btn-danger' : 'btn-primary'}`}
                style={{
                  width: '100%',
                  padding: '15px',
                  fontSize: '1.2em',
                  background: isRecording ? '#dc3545' : '#0d6efd',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                {isRecording ? '⏹ Detener Grabación' : '⏺ Iniciar Grabación'}
              </button>

              {!isRecording && samplesCount > 0 && (
                <button
                  onClick={() => {
                    setNombre('')
                    setSamplesCount(0)
                    setEstado('')
                    setLastCapture(null)
                  }}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer'
                  }}
                >
                  ✨ Grabar Otra Seña (Limpiar)
                </button>
              )}
            </div>

            {estado && (
              <div style={{ marginTop: '15px', padding: '10px', background: estado.includes('✅') ? '#d1e7dd' : '#f8d7da', color: estado.includes('✅') ? '#0f5132' : '#842029', borderRadius: '5px', fontSize: '1em', fontWeight: 'bold', textAlign: 'center' }}>
                {estado}
              </div>
            )}

            {lastCapture && (
              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <p style={{ fontWeight: 'bold', marginBottom: '5px' }}>👁️ Así lo ve la IA (Segmentado):</p>
                <img src={lastCapture} alt="Última muestra" style={{ width: '100%', borderRadius: '8px', border: '2px solid #0d6efd', background: '#000' }} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
