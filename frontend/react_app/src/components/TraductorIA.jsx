import React, { useState, useRef, useEffect } from 'react'
import api from '../services/api'
import './TraductorIA.css'

export default function TraductorIA() {
    // Estados principales
    const [isActive, setIsActive] = useState(false)
    const [mode, setMode] = useState('traduccion') // Empezar directamente en traducción
    const [lastPrediction, setLastPrediction] = useState(null)
    const [aiContext, setAiContext] = useState("Iniciando cámara...")
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [modelLoaded, setModelLoaded] = useState(false)
    const [modelGestures, setModelGestures] = useState([])

    // Para traducción natural
    const [gestureBuffer, setGestureBuffer] = useState([])
    const [naturalTranslation, setNaturalTranslation] = useState("")
    const [isProcessingBuffer, setIsProcessingBuffer] = useState(false)

    // Debug info
    const [debugInfo, setDebugInfo] = useState({ handsDetected: 0, confidence: 0, rawGesture: '' })

    // Refs
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const streamRef = useRef(null)
    const intervalRef = useRef(null)

    // Refs para estabilidad
    const stabilityCounter = useRef(0)
    const currentCandidate = useRef(null)
    const lastSpokenGesture = useRef(null)
    const silenceCounter = useRef(0)
    const bufferRef = useRef([])

    // Verificar modelo al iniciar
    useEffect(() => {
        checkModel()
    }, [])

    const checkModel = async () => {
        try {
            const res = await api.get('/recognize/model-info')
            if (res.data.loaded) {
                setModelLoaded(true)
                setModelGestures(res.data.gestures || [])
                setAiContext(`Modelo listo (${res.data.num_gestures} señas)`)
            } else {
                setModelLoaded(false)
                setAiContext("⚠️ Modelo no entrenado. Entrena el modelo primero.")
            }
        } catch (e) {
            console.error("Error verificando modelo:", e)
            setModelLoaded(false)
            setAiContext("❌ Error conectando con el servidor")
        }
    }

    // Síntesis de voz
    const speak = (text) => {
        if (!text) return
        window.speechSynthesis.cancel()
        setIsSpeaking(false)

        const utterance = new SpeechSynthesisUtterance(text)
        utterance.lang = 'es-ES'
        utterance.rate = 0.8

        utterance.onstart = () => setIsSpeaking(true)
        utterance.onend = () => setIsSpeaking(false)
        utterance.onerror = () => setIsSpeaking(false)

        window.speechSynthesis.speak(utterance)
    }

    // Iniciar cámara
    useEffect(() => {
        startCamera()
        return () => {
            stopCamera()
        }
    }, [])

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: 30 }
            })
            streamRef.current = stream
            if (videoRef.current) {
                videoRef.current.srcObject = stream
                videoRef.current.onloadedmetadata = () => {
                    setIsActive(true)
                }
            }
        } catch (err) {
            console.error("Error cámara:", err)
            setAiContext("❌ Error: No se pudo acceder a la cámara")
        }
    }

    const stopCamera = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
        }
        setIsActive(false)
    }

    // Bucle de procesamiento con interval (más confiable que requestAnimationFrame)
    useEffect(() => {
        if (isActive && mode === 'traduccion') {
            intervalRef.current = setInterval(processFrame, 150) // ~6-7 fps
        } else if (isActive && mode === 'exploracion') {
            intervalRef.current = setInterval(processFrameExploration, 200)
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current)
            }
        }
    }, [isActive, mode])

    const processFrame = async () => {
        if (!videoRef.current || !canvasRef.current) return

        const canvas = document.createElement('canvas')
        canvas.width = videoRef.current.videoWidth || 640
        canvas.height = videoRef.current.videoHeight || 480
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

        canvas.toBlob(async (blob) => {
            if (!blob) return

            const formData = new FormData()
            formData.append('image', blob, 'frame.jpg')

            try {
                const res = await api.post('/recognize/predict?use_llm=false', formData, {
                    timeout: 3000
                })

                const data = res.data

                // Actualizar debug info
                setDebugInfo({
                    handsDetected: data.num_hands || 0,
                    confidence: data.confidence ? Math.round(data.confidence * 100) : 0,
                    rawGesture: data.gesture || '-'
                })

                // Dibujar landmarks si hay
                if (data.hands && data.hands.length > 0) {
                    drawLandmarks(data.hands)
                    // Procesar predicción SOLO SI HAY MANOS DETECTADAS
                    handlePrediction(data)
                } else {
                    // Si no hay manos, limpiar estado de predicción
                    clearCanvas()
                    stabilityCounter.current = 0
                    currentCandidate.current = null
                    setAiContext("Esperando manos...")
                }

            } catch (err) {
                // Solo detectar manos si predict falla
                try {
                    const res2 = await api.post('/capture/process-landmarks', formData)
                    if (res2.data.hands_detected > 0) {
                        setDebugInfo({ handsDetected: res2.data.hands_detected, confidence: 0, rawGesture: '(modelo no disponible)' })
                        setAiContext(`Veo ${res2.data.hands_detected} mano(s) - Modelo no cargado`)
                    }
                } catch (e2) {
                    // Silenciar
                }
            }
        }, 'image/jpeg', 0.7)
    }

    const processFrameExploration = async () => {
        if (!videoRef.current) return

        const canvas = document.createElement('canvas')
        canvas.width = videoRef.current.videoWidth || 640
        canvas.height = videoRef.current.videoHeight || 480
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

        canvas.toBlob(async (blob) => {
            if (!blob) return
            const formData = new FormData()
            formData.append('image', blob, 'frame.jpg')

            try {
                const res = await api.post('/capture/process-landmarks', formData)
                const data = res.data

                setDebugInfo({
                    handsDetected: data.hands_detected || 0,
                    confidence: 0,
                    rawGesture: '-'
                })

                if (data.hands && data.hands.length > 0) {
                    drawLandmarks(data.hands)
                    setAiContext(`Detectadas ${data.hands_detected} mano(s)`)
                } else {
                    clearCanvas()
                    setAiContext("Buscando manos...")
                }
            } catch (e) {
                // Silenciar
            }
        }, 'image/jpeg', 0.7)
    }

    const drawLandmarks = (hands) => {
        const canvas = canvasRef.current
        if (!canvas || !videoRef.current) return

        const ctx = canvas.getContext('2d')
        canvas.width = videoRef.current.videoWidth || 640
        canvas.height = videoRef.current.videoHeight || 480
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        hands.forEach(hand => {
            const landmarks = hand.landmarks
            if (!landmarks) return

            // Dibujar puntos
            ctx.fillStyle = '#00FF00'
            landmarks.forEach(point => {
                const x = point[0] * canvas.width
                const y = point[1] * canvas.height
                ctx.beginPath()
                ctx.arc(x, y, 5, 0, 2 * Math.PI)
                ctx.fill()
            })

            // Dibujar conexiones
            ctx.strokeStyle = '#00FF00'
            ctx.lineWidth = 2
            const connections = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [5, 9, 10, 11, 12],
                [9, 13, 14, 15, 16],
                [13, 17, 18, 19, 20],
                [0, 17]
            ]
            connections.forEach(conn => {
                ctx.beginPath()
                ctx.moveTo(landmarks[conn[0]][0] * canvas.width, landmarks[conn[0]][1] * canvas.height)
                for (let i = 1; i < conn.length; i++) {
                    ctx.lineTo(landmarks[conn[i]][0] * canvas.width, landmarks[conn[i]][1] * canvas.height)
                }
                ctx.stroke()
            })
        })
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current
        if (canvas) {
            const ctx = canvas.getContext('2d')
            ctx.clearRect(0, 0, canvas.width, canvas.height)
        }
    }

    const handlePrediction = (data) => {
        const gesture = data.gesture
        const confidence = data.confidence || 0

        // Sin detección o baja confianza
        if (!gesture || confidence < 0.4) {
            stabilityCounter.current = 0
            currentCandidate.current = null
            silenceCounter.current += 1

            if (silenceCounter.current > 10) {
                lastSpokenGesture.current = null
            }

            // Auto-traducir tras pausa
            if (silenceCounter.current === 25 && bufferRef.current.length > 0) {
                synthesizeSentence()
            } else if (silenceCounter.current > 25 && bufferRef.current.length === 0) {
                setAiContext("Esperando señas...")
            }
            return
        }

        silenceCounter.current = 0

        // Estabilidad de detección
        if (gesture === currentCandidate.current) {
            stabilityCounter.current += 1
        } else {
            currentCandidate.current = gesture
            stabilityCounter.current = 1
        }

        // Mostrar lo que ve (baja confianza o inestable)
        // Subimos el umbral de silencio a 0.8 (80%) para evitar que "hable por hablar"
        if (confidence < 0.8) {
            setAiContext(`Analizando...`)
            // Si la confianza es baja, reseteamos la estabilidad para no acumular falsos positivos
            stabilityCounter.current = Math.max(0, stabilityCounter.current - 1)
            return
        }

        // Captura confirmada
        // Aumentamos la estabilidad requerida a 8 frames consecutivos (aprox 1.2 segundos de pose mantenida)
        // Esto es CLAVE para que no dispare traduciones mientras te mueves de una pose a otra
        if (stabilityCounter.current >= 8 && lastSpokenGesture.current !== gesture) {
            lastSpokenGesture.current = gesture
            setLastPrediction(gesture)

            bufferRef.current = [...bufferRef.current, gesture]
            setGestureBuffer([...bufferRef.current])
            setAiContext(`✅ ${gesture}`)

            // Feedback de voz inmediato
            speak(gesture)
        }
    }

    const synthesizeSentence = async () => {
        if (bufferRef.current.length === 0 || isProcessingBuffer) return

        setIsProcessingBuffer(true)
        setAiContext("✨ Traduciendo frase...")

        try {
            const res = await api.post('/recognize/translate-sequence', {
                gestures: bufferRef.current
            })

            if (res.data.success) {
                const translation = res.data.translation
                setNaturalTranslation(translation)
                setAiContext("¡Traducido!")
                speak(translation)

                bufferRef.current = []
                setGestureBuffer([])
                lastSpokenGesture.current = null
            } else {
                throw new Error("Traducción fallida")
            }
        } catch (err) {
            // Fallback local
            const words = bufferRef.current.join(" ")
            const fallback = words.charAt(0).toUpperCase() + words.slice(1) + "."
            setNaturalTranslation(fallback)
            setAiContext("Traducción local")
            speak(fallback)

            bufferRef.current = []
            setGestureBuffer([])
        } finally {
            setIsProcessingBuffer(false)
        }
    }

    const resetBuffer = () => {
        bufferRef.current = []
        setGestureBuffer([])
        setNaturalTranslation("")
        lastSpokenGesture.current = null
        setAiContext("Buffer limpiado")
    }

    const toggleMode = () => {
        setMode(m => m === 'traduccion' ? 'exploracion' : 'traduccion')
        resetBuffer()
    }

    return (
        <div className="traductor-container" style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h2 style={{ marginBottom: '20px' }}>🤟 Traductor de Lengua de Señas</h2>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '24px' }}>
                {/* Video */}
                <div style={{ position: 'relative', background: '#000', borderRadius: '12px', overflow: 'hidden' }}>
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{ width: '100%', display: 'block' }}
                    />
                    <canvas
                        ref={canvasRef}
                        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                    />

                    {/* Estado */}
                    <div style={{
                        position: 'absolute',
                        top: 10,
                        left: 10,
                        background: isActive ? 'rgba(76,175,80,0.9)' : 'rgba(244,67,54,0.9)',
                        color: 'white',
                        padding: '8px 14px',
                        borderRadius: '20px',
                        fontSize: '0.9em',
                        fontWeight: '600'
                    }}>
                        {isActive ? '● Activo' : '○ Inactivo'}
                    </div>

                    {/* Modo */}
                    <div style={{
                        position: 'absolute',
                        top: 10,
                        right: 10,
                        background: mode === 'traduccion' ? 'rgba(103,58,183,0.9)' : 'rgba(33,150,243,0.9)',
                        color: 'white',
                        padding: '8px 14px',
                        borderRadius: '20px',
                        fontSize: '0.9em',
                        fontWeight: '600'
                    }}>
                        {mode === 'traduccion' ? '🎯 Traduciendo' : '👁️ Explorando'}
                    </div>

                    {/* Debug overlay */}
                    <div style={{
                        position: 'absolute',
                        bottom: 10,
                        left: 10,
                        background: 'rgba(0,0,0,0.7)',
                        color: '#0f0',
                        padding: '10px',
                        borderRadius: '8px',
                        fontFamily: 'monospace',
                        fontSize: '0.85em'
                    }}>
                        Manos: {debugInfo.handsDetected} | Conf: {debugInfo.confidence}% | Seña: {debugInfo.rawGesture}
                    </div>

                    {/* Última predicción grande */}
                    {lastPrediction && mode === 'traduccion' && (
                        <div style={{
                            position: 'absolute',
                            bottom: 60,
                            left: '50%',
                            transform: 'translateX(-50%)',
                            background: 'rgba(103,58,183,0.95)',
                            color: 'white',
                            padding: '15px 30px',
                            borderRadius: '30px',
                            fontSize: '1.8em',
                            fontWeight: 'bold',
                            textTransform: 'uppercase'
                        }}>
                            {lastPrediction}
                        </div>
                    )}
                </div>

                {/* Panel lateral */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {/* Estado del modelo */}
                    <div style={{
                        background: modelLoaded ? '#e8f5e9' : '#ffebee',
                        border: modelLoaded ? '2px solid #4caf50' : '2px solid #f44336',
                        borderRadius: '12px',
                        padding: '16px'
                    }}>
                        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
                            {modelLoaded ? '✅ Modelo Cargado' : '❌ Modelo No Disponible'}
                        </div>
                        {modelLoaded && modelGestures.length > 0 && (
                            <div style={{ fontSize: '0.85em', color: '#666' }}>
                                Señas: {modelGestures.slice(0, 10).join(', ')}{modelGestures.length > 10 ? '...' : ''}
                            </div>
                        )}
                        {!modelLoaded && (
                            <div style={{ fontSize: '0.9em', color: '#c62828' }}>
                                Ve a "Entrenar Modelo" primero
                            </div>
                        )}
                    </div>

                    {/* Contexto IA */}
                    <div style={{
                        background: '#f5f5f5',
                        borderRadius: '12px',
                        padding: '16px'
                    }}>
                        <div style={{ fontWeight: '600', marginBottom: '8px', color: '#666' }}>Estado:</div>
                        <div style={{ fontSize: '1.1em', fontWeight: '500' }}>{aiContext}</div>
                        {isSpeaking && <div style={{ marginTop: '8px', color: '#1976d2' }}>🔊 Hablando...</div>}
                    </div>

                    {/* Buffer de gestos */}
                    {mode === 'traduccion' && (
                        <div style={{
                            background: '#fff',
                            border: '2px solid #e0e0e0',
                            borderRadius: '12px',
                            padding: '16px'
                        }}>
                            <div style={{ fontWeight: '600', marginBottom: '10px' }}>Secuencia capturada:</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', minHeight: '40px' }}>
                                {gestureBuffer.length > 0 ? (
                                    gestureBuffer.map((g, i) => (
                                        <span key={i} style={{
                                            background: '#673ab7',
                                            color: 'white',
                                            padding: '6px 12px',
                                            borderRadius: '15px',
                                            fontSize: '0.9em',
                                            fontWeight: '500'
                                        }}>
                                            {g}
                                        </span>
                                    ))
                                ) : (
                                    <span style={{ color: '#999' }}>Realiza señas para capturar...</span>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Traducción */}
                    {naturalTranslation && (
                        <div style={{
                            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            color: 'white',
                            borderRadius: '12px',
                            padding: '20px'
                        }}>
                            <div style={{ fontWeight: '600', marginBottom: '8px', opacity: 0.9 }}>Traducción:</div>
                            <div style={{ fontSize: '1.3em', fontWeight: '500' }}>{naturalTranslation}</div>
                        </div>
                    )}

                    {/* Controles */}
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <button
                            onClick={toggleMode}
                            style={{
                                flex: 1,
                                padding: '14px',
                                background: mode === 'traduccion' ? '#2196F3' : '#673ab7',
                                color: 'white',
                                border: 'none',
                                borderRadius: '10px',
                                cursor: 'pointer',
                                fontWeight: '600',
                                fontSize: '0.95em'
                            }}
                        >
                            {mode === 'traduccion' ? '👁️ Modo Explorar' : '🎯 Modo Traducir'}
                        </button>

                        {mode === 'traduccion' && (
                            <>
                                <button
                                    onClick={synthesizeSentence}
                                    disabled={gestureBuffer.length === 0 || isProcessingBuffer}
                                    style={{
                                        padding: '14px 20px',
                                        background: gestureBuffer.length > 0 ? '#4caf50' : '#ccc',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '10px',
                                        cursor: gestureBuffer.length > 0 ? 'pointer' : 'not-allowed',
                                        fontWeight: '600'
                                    }}
                                >
                                    ✨
                                </button>
                                <button
                                    onClick={resetBuffer}
                                    style={{
                                        padding: '14px 20px',
                                        background: '#ff5722',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '10px',
                                        cursor: 'pointer',
                                        fontWeight: '600'
                                    }}
                                >
                                    🗑️
                                </button>
                            </>
                        )}
                    </div>

                    <button
                        onClick={() => speak("Prueba de audio")}
                        style={{
                            padding: '12px',
                            background: '#607d8b',
                            color: 'white',
                            border: 'none',
                            borderRadius: '10px',
                            cursor: 'pointer',
                            fontWeight: '500'
                        }}
                    >
                        🔊 Probar Audio
                    </button>

                    <button
                        onClick={checkModel}
                        style={{
                            padding: '12px',
                            background: '#9e9e9e',
                            color: 'white',
                            border: 'none',
                            borderRadius: '10px',
                            cursor: 'pointer',
                            fontWeight: '500'
                        }}
                    >
                        🔄 Recargar Modelo
                    </button>
                </div>
            </div>
        </div>
    )
}
