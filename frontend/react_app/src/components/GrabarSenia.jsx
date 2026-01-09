import React, { useState, useRef, useEffect, useCallback } from 'react'
import api from '../services/api'

// Configuración de muestras
const TARGET_SAMPLES = 40
const SEQUENCE_FRAMES = 30  // Frames por secuencia para señas dinámicas
const SEQUENCE_INTERVAL_MS = 100  // ~10 fps para captura de secuencia

export default function GrabarSenia() {
  const [nombre, setNombre] = useState('')
  const [estado, setEstado] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [samplesCount, setSamplesCount] = useState(0)
  const [lastCapture, setLastCapture] = useState(null)

  // Modo de captura (estático vs dinámico)
  const [captureMode, setCaptureMode] = useState('static') // 'static' | 'dynamic'

  // Para captura dinámica continua
  const [currentSequenceFrame, setCurrentSequenceFrame] = useState(0)
  const sequenceBufferRef = useRef([])

  // Progreso visual
  const progress = Math.min((samplesCount / TARGET_SAMPLES) * 100, 100)
  const isComplete = samplesCount >= TARGET_SAMPLES

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const recordingIntervalRef = useRef(null)
  const samplesCountRef = useRef(0)

  // Sync ref con state
  useEffect(() => {
    samplesCountRef.current = samplesCount
  }, [samplesCount])

  // Iniciar cámara al montar
  useEffect(() => {
    startCamera()
    return () => {
      stopCamera()
      stopRecording()
    }
  }, [])

  // Auto-detener cuando se completan las muestras
  useEffect(() => {
    if (isComplete && isRecording) {
      stopRecording()
      setEstado(`🎉 ¡Completado! ${TARGET_SAMPLES} muestras guardadas para "${nombre}"`)

      try {
        const utterance = new SpeechSynthesisUtterance('Grabación completada')
        utterance.lang = 'es-ES'
        utterance.rate = 1.2
        window.speechSynthesis.speak(utterance)
      } catch (e) { /* Silenciar errores */ }
    }
  }, [isComplete, isRecording, nombre])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 30 }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setEstado('📸 Cámara activa. Selecciona el modo y comienza a grabar.')
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

  const stopRecording = () => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current)
      recordingIntervalRef.current = null
    }
    setIsRecording(false)
  }

  const flashBorder = (color) => {
    const video = videoRef.current
    if (video) {
      video.style.border = `4px solid ${color}`
      setTimeout(() => {
        video.style.border = 'none'
      }, 200)
    }
  }

  const captureCurrentFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    return new Promise(resolve => {
      canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.85)
    })
  }

  // ============= CAPTURA ESTÁTICA =============
  const captureStaticFrame = async () => {
    if (samplesCountRef.current >= TARGET_SAMPLES) {
      stopRecording()
      return
    }

    const blob = await captureCurrentFrame()
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
        setLastCapture(res.data.segmented_image || URL.createObjectURL(blob))
        setEstado(`✅ Muestra ${samplesCountRef.current + 1}/${TARGET_SAMPLES}`)
        flashBorder('#4caf50')
      } else {
        setEstado('🔎 Buscando manos...')
      }
    } catch (err) {
      console.error('Error:', err)
      setEstado('⚠️ Error de conexión')
    }
  }

  // ============= CAPTURA DINÁMICA CONTINUA =============
  const captureDynamicFrame = async () => {
    if (samplesCountRef.current >= TARGET_SAMPLES) {
      stopRecording()
      return
    }

    const blob = await captureCurrentFrame()
    if (!blob) return

    // Agregar frame al buffer
    sequenceBufferRef.current.push(blob)
    setCurrentSequenceFrame(sequenceBufferRef.current.length)

    // Cuando completamos una secuencia
    if (sequenceBufferRef.current.length >= SEQUENCE_FRAMES) {
      setEstado(`📤 Enviando secuencia ${samplesCountRef.current + 1}...`)

      const frames = [...sequenceBufferRef.current]
      sequenceBufferRef.current = []  // Reset buffer para la siguiente secuencia
      setCurrentSequenceFrame(0)

      // Enviar secuencia al backend
      await sendSequence(frames)
    } else {
      setEstado(`🎬 Capturando: ${sequenceBufferRef.current.length}/${SEQUENCE_FRAMES} frames`)
    }
  }

  const sendSequence = async (frames) => {
    try {
      const formData = new FormData()
      formData.append('gesture_name', nombre)
      formData.append('sequence_length', frames.length.toString())

      frames.forEach((blob, index) => {
        formData.append(`frame_${index}`, blob, `frame_${index}.jpg`)
      })

      const res = await api.post('/gestures/capture-sequence', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000
      })

      if (res.data.success) {
        setSamplesCount(prev => prev + 1)
        setEstado(`✅ Secuencia ${samplesCountRef.current + 1}/${TARGET_SAMPLES} guardada`)
        if (res.data.preview_image) {
          setLastCapture(res.data.preview_image)
        }
        flashBorder('#9c27b0')
      } else {
        setEstado(`⚠️ ${res.data.error || 'Secuencia inválida - sigue grabando'}`)
        flashBorder('#ff9800')
      }
    } catch (err) {
      console.error('Error enviando secuencia:', err)
      setEstado('⚠️ Error al enviar - continuando...')
    }
  }

  // ============= CONTROL DE GRABACIÓN =============
  const startRecording = () => {
    if (!nombre.trim()) {
      setEstado('⚠️ Escribe un nombre para la seña primero')
      return
    }

    setIsRecording(true)
    sequenceBufferRef.current = []
    setCurrentSequenceFrame(0)

    if (captureMode === 'static') {
      setEstado('🎥 Grabando... Mantén la posición de la seña')
      recordingIntervalRef.current = setInterval(captureStaticFrame, 300)
    } else {
      setEstado('🎬 Grabando secuencias... Realiza el movimiento repetidamente')
      recordingIntervalRef.current = setInterval(captureDynamicFrame, SEQUENCE_INTERVAL_MS)
    }
  }

  const toggleRecording = (e) => {
    e.preventDefault()

    if (isRecording) {
      stopRecording()
      setEstado('⏸️ Grabación pausada')
    } else {
      startRecording()
    }
  }

  const resetCapture = () => {
    stopRecording()
    setNombre('')
    setSamplesCount(0)
    setEstado('📸 Listo para grabar una nueva seña')
    setLastCapture(null)
    sequenceBufferRef.current = []
    setCurrentSequenceFrame(0)
  }

  // Calcular progreso de la secuencia actual (para modo dinámico)
  const sequenceProgress = captureMode === 'dynamic'
    ? (currentSequenceFrame / SEQUENCE_FRAMES) * 100
    : 0

  return (
    <div className="admin-content">
      <div className="container">
        <h2>📹 Grabar Nueva Seña</h2>

        <div className="recording-layout" style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px' }}>

          {/* Columna Izquierda: Cámara */}
          <div className="camera-section">
            <div className="video-container" style={{
              position: 'relative',
              background: '#000',
              borderRadius: '12px',
              overflow: 'hidden',
              minHeight: '480px'
            }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', transition: 'border 0.15s' }}
              />

              {/* Indicador de grabación */}
              {isRecording && (
                <div style={{
                  position: 'absolute',
                  top: 12,
                  right: 12,
                  background: captureMode === 'dynamic' ? '#9c27b0' : '#f44336',
                  color: 'white',
                  padding: '8px 16px',
                  borderRadius: '20px',
                  animation: 'pulse 1s infinite',
                  fontWeight: 'bold',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  boxShadow: '0 2px 10px rgba(0,0,0,0.3)'
                }}>
                  {captureMode === 'dynamic' ? '🎬 GRABANDO MOVIMIENTO' : '🔴 GRABANDO'}
                </div>
              )}

              {/* Badge de modo */}
              <div style={{
                position: 'absolute',
                top: 12,
                left: 12,
                background: captureMode === 'static' ? 'rgba(33, 150, 243, 0.95)' : 'rgba(156, 39, 176, 0.95)',
                color: 'white',
                padding: '8px 14px',
                borderRadius: '15px',
                fontSize: '0.9em',
                fontWeight: '600',
                boxShadow: '0 2px 10px rgba(0,0,0,0.2)'
              }}>
                {captureMode === 'static' ? '📷 Modo Estático' : '🎬 Modo Dinámico'}
              </div>

              {/* Barra de progreso de secuencia (solo modo dinámico) */}
              {captureMode === 'dynamic' && isRecording && (
                <div style={{
                  position: 'absolute',
                  bottom: 50,
                  left: '10%',
                  right: '10%',
                  background: 'rgba(0,0,0,0.7)',
                  borderRadius: '10px',
                  padding: '8px 12px'
                }}>
                  <div style={{ color: 'white', fontSize: '0.85em', marginBottom: '4px', textAlign: 'center' }}>
                    Secuencia actual: {currentSequenceFrame}/{SEQUENCE_FRAMES}
                  </div>
                  <div style={{ height: '6px', background: 'rgba(255,255,255,0.3)', borderRadius: '3px' }}>
                    <div style={{
                      height: '100%',
                      width: `${sequenceProgress}%`,
                      background: '#9c27b0',
                      borderRadius: '3px',
                      transition: 'width 0.1s'
                    }} />
                  </div>
                </div>
              )}

              {/* Barra de progreso total */}
              {samplesCount > 0 && (
                <div style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: '8px',
                  background: 'rgba(0,0,0,0.5)'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${progress}%`,
                    background: isComplete ? '#4caf50' : 'linear-gradient(90deg, #2196F3, #21CBF3)',
                    transition: 'width 0.3s ease-out'
                  }} />
                </div>
              )}
            </div>
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Instrucciones según modo */}
            <div style={{
              marginTop: '12px',
              padding: '14px 16px',
              background: captureMode === 'static' ? '#e3f2fd' : '#f3e5f5',
              borderRadius: '10px',
              fontSize: '0.95em',
              lineHeight: '1.5'
            }}>
              {captureMode === 'static' ? (
                <>
                  <strong>📷 Modo Estático:</strong> Para señas que son una posición fija (letras, números).
                  <br />Mantén la posición mientras se capturan las muestras automáticamente.
                </>
              ) : (
                <>
                  <strong>🎬 Modo Dinámico:</strong> Para señas con movimiento (saludar, despedir, etc).
                  <br />Repite el movimiento continuamente. Cada {SEQUENCE_FRAMES} frames se guarda automáticamente una secuencia.
                  <br /><em style={{ color: '#7b1fa2' }}>Solo inicia y sigue haciendo el movimiento hasta completar.</em>
                </>
              )}
            </div>
          </div>

          {/* Columna Derecha: Controles */}
          <div className="controls-section">
            {/* Nombre de la seña */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '6px', fontWeight: '600', color: '#333' }}>
                Nombre de la Seña
              </label>
              <input
                type="text"
                value={nombre}
                onChange={e => setNombre(e.target.value.toLowerCase().replace(/\s+/g, '_'))}
                placeholder="Ej: hola, gracias, 1, a..."
                disabled={isRecording}
                style={{
                  width: '100%',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  border: '2px solid #e0e0e0',
                  fontSize: '1em',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            {/* Selector de modo */}
            <div style={{
              display: 'flex',
              gap: '8px',
              marginBottom: '16px',
              background: '#f0f0f0',
              padding: '5px',
              borderRadius: '12px'
            }}>
              <button
                onClick={() => setCaptureMode('static')}
                disabled={isRecording}
                style={{
                  flex: 1,
                  padding: '12px',
                  border: 'none',
                  borderRadius: '10px',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  background: captureMode === 'static' ? '#2196F3' : 'transparent',
                  color: captureMode === 'static' ? 'white' : '#666',
                  transition: 'all 0.2s',
                  fontSize: '0.95em'
                }}
              >
                📷 Estático
              </button>
              <button
                onClick={() => setCaptureMode('dynamic')}
                disabled={isRecording}
                style={{
                  flex: 1,
                  padding: '12px',
                  border: 'none',
                  borderRadius: '10px',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  background: captureMode === 'dynamic' ? '#9c27b0' : 'transparent',
                  color: captureMode === 'dynamic' ? 'white' : '#666',
                  transition: 'all 0.2s',
                  fontSize: '0.95em'
                }}
              >
                🎬 Dinámico
              </button>
            </div>

            {/* Progreso */}
            <div style={{
              background: isComplete ? '#e8f5e9' : '#f5f5f5',
              padding: '18px',
              borderRadius: '14px',
              marginBottom: '16px',
              textAlign: 'center',
              border: isComplete ? '2px solid #4caf50' : '2px solid transparent',
              transition: 'all 0.3s'
            }}>
              <div style={{
                fontSize: '2.5em',
                fontWeight: 'bold',
                color: isComplete ? '#4caf50' : '#333',
                marginBottom: '8px'
              }}>
                {samplesCount} / {TARGET_SAMPLES}
              </div>
              <div style={{
                width: '100%',
                height: '10px',
                background: '#e0e0e0',
                borderRadius: '5px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${progress}%`,
                  background: isComplete
                    ? '#4caf50'
                    : captureMode === 'static'
                      ? 'linear-gradient(90deg, #2196F3, #21CBF3)'
                      : 'linear-gradient(90deg, #9c27b0, #e040fb)',
                  borderRadius: '5px',
                  transition: 'width 0.3s ease-out'
                }} />
              </div>
              <p style={{
                fontSize: '0.9em',
                color: isComplete ? '#2e7d32' : '#666',
                marginTop: '10px',
                fontWeight: isComplete ? '600' : 'normal'
              }}>
                {isComplete
                  ? '✅ ¡Completo! Puedes entrenar el modelo.'
                  : captureMode === 'static'
                    ? `Faltan ${TARGET_SAMPLES - samplesCount} muestras`
                    : `Faltan ${TARGET_SAMPLES - samplesCount} secuencias`}
              </p>
            </div>

            {/* Botones de acción */}
            <div style={{ display: 'flex', gap: '10px', flexDirection: 'column' }}>
              <button
                onClick={toggleRecording}
                disabled={isComplete || !nombre.trim()}
                style={{
                  width: '100%',
                  padding: '16px',
                  fontSize: '1.15em',
                  fontWeight: 'bold',
                  background: isComplete
                    ? '#9e9e9e'
                    : isRecording
                      ? '#f44336'
                      : captureMode === 'static' ? '#2196F3' : '#9c27b0',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: (isComplete || !nombre.trim()) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  opacity: !nombre.trim() ? 0.6 : 1,
                  boxShadow: isRecording ? '0 4px 15px rgba(244,67,54,0.4)' : '0 4px 15px rgba(0,0,0,0.1)'
                }}
              >
                {isComplete
                  ? '✅ Completado'
                  : isRecording
                    ? '⏹ Detener Grabación'
                    : captureMode === 'static'
                      ? '⏺ Iniciar Grabación'
                      : '🎬 Iniciar Grabación Continua'
                }
              </button>

              {samplesCount > 0 && !isRecording && (
                <button
                  onClick={resetCapture}
                  style={{
                    width: '100%',
                    padding: '14px',
                    background: '#607d8b',
                    color: 'white',
                    border: 'none',
                    borderRadius: '10px',
                    cursor: 'pointer',
                    fontWeight: '600',
                    fontSize: '1em'
                  }}
                >
                  🔄 Grabar Otra Seña
                </button>
              )}
            </div>

            {/* Estado */}
            {estado && (
              <div style={{
                marginTop: '16px',
                padding: '14px',
                background: estado.includes('✅') || estado.includes('🎉')
                  ? '#e8f5e9'
                  : estado.includes('❌')
                    ? '#ffebee'
                    : estado.includes('⚠️')
                      ? '#fff3e0'
                      : '#e3f2fd',
                color: estado.includes('✅') || estado.includes('🎉')
                  ? '#2e7d32'
                  : estado.includes('❌')
                    ? '#c62828'
                    : estado.includes('⚠️')
                      ? '#e65100'
                      : '#1565c0',
                borderRadius: '10px',
                fontSize: '0.95em',
                fontWeight: '600',
                textAlign: 'center'
              }}>
                {estado}
              </div>
            )}

            {/* Preview */}
            {lastCapture && (
              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <p style={{ fontWeight: '600', marginBottom: '10px', color: '#555' }}>
                  👁️ Última captura:
                </p>
                <img
                  src={lastCapture}
                  alt="Última muestra"
                  style={{
                    width: '100%',
                    borderRadius: '12px',
                    border: `3px solid ${captureMode === 'static' ? '#2196F3' : '#9c27b0'}`,
                    background: '#000'
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.02); }
        }
      `}</style>
    </div>
  )
}
