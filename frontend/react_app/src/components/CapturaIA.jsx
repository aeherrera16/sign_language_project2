import React, { useRef, useEffect, useState, useCallback } from 'react'
import api from '../services/api'
import '../styles/admin.css'

export default function CapturaIA() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsCanvasRef = useRef(null)  // Canvas para mostrar solo manos

  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [gestureName, setGestureName] = useState('')
  const [saving, setSaving] = useState(false)
  const [mode, setMode] = useState('hands-focus')  // 'hands-focus' | 'body' | 'landmarks'
  const [handsUrl, setHandsUrl] = useState(null)

  const qualityThreshold = 60

  // Inicializar cámara
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }
    })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      })
      .catch(err => console.error('Cámara no disponible:', err))

    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  // Loop de análisis
  useEffect(() => {
    if (!analyzing) return
    const id = setInterval(() => {
      analyzeFrame()
    }, 1500)  // Más rápido para mejor feedback
    return () => clearInterval(id)
  }, [analyzing, mode])

  const getFrameBlob = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    const targetW = Math.max(480, video.videoWidth)
    const targetH = Math.max(360, video.videoHeight)
    canvas.width = targetW
    canvas.height = targetH

    const ctx = canvas.getContext('2d')
    ctx.filter = 'contrast(120%) saturate(110%)'
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    ctx.filter = 'none'

    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9))
  }, [])

  const analyzeFrame = async () => {
    const blob = await getFrameBlob()
    if (!blob) return

    const fd = new FormData()
    fd.append('file', blob, 'frame.jpg')

    try {
      // Elegir endpoint según modo
      let endpoint = '/api/capture/hands-focus'
      if (mode === 'body') {
        endpoint = '/api/capture/segment-frame'
      } else if (mode === 'landmarks') {
        endpoint = '/api/capture/hands-landmarks'
      }

      const segRes = await fetch(api.defaults.baseURL.replace('/api', '') + endpoint, {
        method: 'POST',
        body: fd
      })

      if (segRes.ok) {
        const segBlob = await segRes.blob()
        const url = URL.createObjectURL(segBlob)

        // Revocar URL anterior para liberar memoria
        if (handsUrl) URL.revokeObjectURL(handsUrl)
        setHandsUrl(url)

        // Dibujar en canvas de manos
        const handsCanvas = handsCanvasRef.current
        if (handsCanvas) {
          const video = videoRef.current
          handsCanvas.width = Math.max(480, video?.videoWidth || 480)
          handsCanvas.height = Math.max(360, video?.videoHeight || 360)
          const hctx = handsCanvas.getContext('2d')
          const img = new Image()
          img.onload = () => {
            hctx.clearRect(0, 0, handsCanvas.width, handsCanvas.height)
            hctx.drawImage(img, 0, 0, handsCanvas.width, handsCanvas.height)
          }
          img.src = url
        }

        // Obtener métricas del header si disponibles
        const handsDetected = segRes.headers.get('X-Hands-Detected')
        const qualityScore = segRes.headers.get('X-Quality-Score')

        if (handsDetected !== null) {
          // Ahora obtener análisis completo de calidad
          await getQualityAnalysis()
        }
      }
    } catch (e) {
      console.warn('Error en segmentación:', e)
    }
  }

  const getQualityAnalysis = async () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const base64 = canvas.toDataURL('image/jpeg', 0.8)

    try {
      // Usar nuevo endpoint de calidad enfocado en manos
      const res = await api.post('/capture/hands-quality', {
        image_base64: base64,
        gesture_name: gestureName
      })
      setAnalysis(res.data)
    } catch (e) {
      // Fallback al endpoint original
      try {
        const res = await api.post('/capture/analyze-capture', {
          image_base64: base64,
          gesture_name: gestureName
        })
        setAnalysis(res.data)
      } catch (err) {
        console.error('Error analizando:', err)
      }
    }
  }

  const handleCapture = async () => {
    await analyzeFrame()

    if (!analysis) {
      alert('No se pudo analizar la captura')
      return
    }

    if (analysis.is_good && analysis.score >= qualityThreshold) {
      const blob = await getFrameBlob()
      if (!blob) return

      setSaving(true)
      try {
        const form = new FormData()
        form.append('image', blob, `${gestureName || 'unknown'}.jpg`)
        form.append('gesture_id', '')
        form.append('metadata', JSON.stringify({
          gesture_name: gestureName,
          analysis,
          capture_mode: mode
        }))

        const saveRes = await api.post('/capture/save', form, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        if (saveRes.status === 200 || saveRes.status === 201) {
          alert('✅ Captura guardada con ID: ' + (saveRes.data.id || 'OK'))
        } else {
          alert('Error al guardar captura')
        }
      } catch (e) {
        console.error('Error guardando:', e)
        alert('Error guardando la captura')
      } finally {
        setSaving(false)
      }
    } else {
      alert(`❌ Captura no cumple umbral de calidad: ${analysis?.score || 'N/A'}/100\n\n${analysis?.recommendations?.join('\n') || ''}`)
    }
  }

  const getQualityColor = (score) => {
    if (score >= 80) return '#22c55e'  // Verde
    if (score >= 60) return '#eab308'  // Amarillo
    if (score >= 40) return '#f97316'  // Naranja
    return '#ef4444'  // Rojo
  }

  const getQualityEmoji = (quality) => {
    const emojis = {
      'excelente': '🌟',
      'buena': '✅',
      'regular': '⚠️',
      'mala': '❌'
    }
    return emojis[quality] || '❓'
  }

  return (
    <div className="captura-ia-view" style={{ padding: '20px' }}>
      <h2 style={{ marginBottom: 16 }}>🖐️ Captura Inteligente de Señas</h2>

      {/* Selector de modo */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 8 }}>
        <button
          className={`btn ${mode === 'hands-focus' ? 'primary' : 'outline'}`}
          onClick={() => setMode('hands-focus')}
        >
          🖐️ Solo Manos
        </button>
        <button
          className={`btn ${mode === 'landmarks' ? 'primary' : 'outline'}`}
          onClick={() => setMode('landmarks')}
        >
          🦴 Landmarks
        </button>
        <button
          className={`btn ${mode === 'body' ? 'primary' : 'outline'}`}
          onClick={() => setMode('body')}
        >
          👤 Cuerpo Completo
        </button>
      </div>

      {/* Vista lado a lado */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 16,
        marginBottom: 16
      }}>
        {/* Video original */}
        <div style={{ position: 'relative' }}>
          <div style={{
            fontSize: 12,
            fontWeight: 600,
            marginBottom: 4,
            color: '#64748b'
          }}>
            📹 Video Original
          </div>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              borderRadius: 12,
              border: '2px solid #e2e8f0'
            }}
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Vista procesada (solo manos) */}
        <div>
          <div style={{
            fontSize: 12,
            fontWeight: 600,
            marginBottom: 4,
            color: '#64748b'
          }}>
            {mode === 'hands-focus' && '🖐️ Solo Manos (Segmentado)'}
            {mode === 'landmarks' && '🦴 Landmarks Detectados'}
            {mode === 'body' && '👤 Segmentación Completa'}
          </div>
          <canvas
            ref={handsCanvasRef}
            style={{
              width: '100%',
              borderRadius: 12,
              border: '2px solid #3b82f6',
              backgroundColor: '#1e293b'
            }}
          />
        </div>
      </div>

      {/* Panel de control */}
      <div style={{
        background: '#f8fafc',
        padding: 16,
        borderRadius: 12,
        border: '1px solid #e2e8f0'
      }}>
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontWeight: 600, marginRight: 8 }}>Nombre de la seña:</label>
          <input
            value={gestureName}
            onChange={(e) => setGestureName(e.target.value)}
            placeholder="Ej: HOLA, GRACIAS, BUENOS_DIAS"
            style={{
              padding: '8px 12px',
              borderRadius: 8,
              border: '1px solid #cbd5e1',
              width: 240
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button
            className="btn primary"
            onClick={() => setAnalyzing(!analyzing)}
            style={{
              background: analyzing ? '#ef4444' : '#22c55e',
              padding: '10px 20px',
              fontSize: 16
            }}
          >
            {analyzing ? '⏹️ Detener Análisis' : '▶️ Iniciar Análisis'}
          </button>
          <button
            className="btn outline"
            onClick={handleCapture}
            disabled={saving || !analysis?.is_good}
            style={{
              padding: '10px 20px',
              fontSize: 16,
              opacity: (saving || !analysis?.is_good) ? 0.5 : 1
            }}
          >
            {saving ? '⏳ Guardando...' : '📸 Capturar y Guardar'}
          </button>
        </div>

        {/* Panel de análisis */}
        {analysis && (
          <div style={{
            background: 'white',
            padding: 16,
            borderRadius: 8,
            border: `2px solid ${getQualityColor(analysis.score)}`
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 12
            }}>
              <h4 style={{ margin: 0 }}>
                {getQualityEmoji(analysis.quality)} Calidad: {analysis.quality?.toUpperCase()}
              </h4>
              <div style={{
                background: getQualityColor(analysis.score),
                color: 'white',
                padding: '4px 12px',
                borderRadius: 20,
                fontWeight: 700,
                fontSize: 18
              }}>
                {analysis.score}/100
              </div>
            </div>

            {/* Barra de progreso */}
            <div style={{
              width: '100%',
              height: 8,
              background: '#e2e8f0',
              borderRadius: 4,
              marginBottom: 12,
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${analysis.score}%`,
                height: '100%',
                background: getQualityColor(analysis.score),
                transition: 'width 0.3s ease'
              }} />
            </div>

            {/* Métricas detalladas */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 8,
              marginBottom: 12
            }}>
              <div style={{
                background: '#f1f5f9',
                padding: 8,
                borderRadius: 6,
                textAlign: 'center'
              }}>
                <div style={{ fontSize: 11, color: '#64748b' }}>Manos</div>
                <div style={{ fontWeight: 700 }}>{analysis.metrics?.hands_detected || 0}</div>
              </div>
              <div style={{
                background: '#f1f5f9',
                padding: 8,
                borderRadius: 6,
                textAlign: 'center'
              }}>
                <div style={{ fontSize: 11, color: '#64748b' }}>Cobertura</div>
                <div style={{ fontWeight: 700 }}>{analysis.metrics?.hands_percentage?.toFixed(1) || 0}%</div>
              </div>
              <div style={{
                background: '#f1f5f9',
                padding: 8,
                borderRadius: 6,
                textAlign: 'center'
              }}>
                <div style={{ fontSize: 11, color: '#64748b' }}>Nitidez</div>
                <div style={{ fontWeight: 700 }}>{analysis.metrics?.sharpness_score?.toFixed(0) || analysis.metrics?.blur_score?.toFixed(0) || 0}</div>
              </div>
            </div>

            {/* Recomendaciones */}
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, color: '#64748b' }}>
                Recomendaciones:
              </div>
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
                {analysis.recommendations?.map((r, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>{r}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Instrucciones cuando no hay análisis */}
        {!analysis && !analyzing && (
          <div style={{
            background: '#eff6ff',
            padding: 16,
            borderRadius: 8,
            border: '1px solid #bfdbfe'
          }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>📋 Instrucciones:</div>
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: 14 }}>
              <li>Escribe el nombre de la seña que vas a capturar</li>
              <li>Haz clic en "Iniciar Análisis" para comenzar</li>
              <li>Posiciona tus manos frente a la cámara (30-60 cm)</li>
              <li>Espera a que la calidad sea "BUENA" o "EXCELENTE"</li>
              <li>Haz clic en "Capturar y Guardar"</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  )
}
