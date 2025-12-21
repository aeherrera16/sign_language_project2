import React, { useState, useRef, useEffect, useCallback } from 'react'
import api from '../services/api'
import './TraductorIA.css' // Asumimos que crearemos este CSS o usaremos inline styles específicos

export default function TraductorIA() {
    // Estados principales
    const [isActive, setIsActive] = useState(false)
    const [mode, setMode] = useState('exploracion') // 'exploracion' (solo ver) | 'traduccion' (reconocer)
    const [lastPrediction, setLastPrediction] = useState(null)
    const [aiContext, setAiContext] = useState("Esperando señas...")
    const [isSpeaking, setIsSpeaking] = useState(false)

    // NUEVOS: Para traducción natural
    const [gestureBuffer, setGestureBuffer] = useState([])
    const [naturalTranslation, setNaturalTranslation] = useState("")
    const [isProcessingBuffer, setIsProcessingBuffer] = useState(false)

    // Refs para video y canvas
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const streamRef = useRef(null)
    const loopRef = useRef(null)

    // Ref para controlar frecuencia de envío (no saturar red)
    const lastFrameTime = useRef(0)

    // Refs para estabilidad y repetición
    const stabilityCounter = useRef(0)
    const currentCandidate = useRef(null)
    const lastSpokenGesture = useRef(null)
    const silenceCounter = useRef(0) // Para resetear el último hablado tras pausa

    // Ref para el buffer de gestos (evitar problemas de clausura en el loop)
    const bufferRef = useRef([])

    // Inicialización de voz
    const speak = (text) => {
        if (!text) return

        // Cancelar cualquier discurso previo antes de empezar uno nuevo
        window.speechSynthesis.cancel()

        // Reset manual
        setIsSpeaking(false)

        const utterance = new SpeechSynthesisUtterance(text)
        utterance.lang = 'es-ES'
        utterance.rate = 0.75 // Velocidad más lenta para mejor comprensión

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
            if (loopRef.current) cancelAnimationFrame(loopRef.current)
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
                    loopRef.current = requestAnimationFrame(processLoop)
                }
            }
            speak("Sistema visual activo. Listo para traducir.")
        } catch (err) {
            console.error("Error cámara:", err)
            setAiContext("Error: No se pudo acceder a la cámara")
        }
    }

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
        }
        setIsActive(false)
    }

    // Bucle principal de procesamiento
    const processLoop = async (timestamp) => {
        if (!videoRef.current || !canvasRef.current || !isActive) return

        // Aumentamos frecuencia a cada 100ms (10 fps de análisis) para mayor respuesta
        if (timestamp - lastFrameTime.current > 100) {
            lastFrameTime.current = timestamp
            await sendFrameToBackend()
        }

        loopRef.current = requestAnimationFrame(processLoop)
    }

    // Enviar frame al backend
    const sendFrameToBackend = async () => {
        if (!videoRef.current) return

        // Capturar frame en canvas auxiliar (o usar el mismo si es invisible)
        const canvas = document.createElement('canvas')
        canvas.width = videoRef.current.videoWidth
        canvas.height = videoRef.current.videoHeight
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

        canvas.toBlob(async (blob) => {
            if (!blob) return

            const formData = new FormData()
            formData.append('image', blob, 'frame.jpg')

            try {
                // Decidir endpoint según modo
                const endpoint = mode === 'traduccion' ? '/recognize/predict' : '/capture/process-landmarks'

                // Fallback a solo landmarks si predict falla (ej: modelo no cargado)
                let res;
                try {
                    // Desactivamos LLM en predict para que sea instantáneo (60fps)
                    res = await api.post(`${endpoint}?use_llm=false`, formData)
                } catch (e) {
                    if (mode === 'traduccion') {
                        // Intentar obtener al menos los landmarks si la predicción falla
                        res = await api.post('/capture/process-landmarks', formData)
                    } else {
                        throw e
                    }
                }

                const data = res.data

                // Logger para ver qué está pasando por detrás
                if (mode === 'traduccion' && data.gesture) {
                    console.log(`IA Detectó: ${data.gesture} (${Math.round(data.confidence * 100)}%)`);
                }

                // Dibujar resultados (puntos verdes)
                // Ahora tanto /predict como /process-landmarks devuelven 'hands'
                drawResults(data)

                // Lógica de traducción y voz
                handleAiReasoning(data)

            } catch (err) {
                console.warn("Error backend loop:", err)
            }
        }, 'image/jpeg', 0.6) // Compresión JPEG 0.6 para velocidad
    }

    // Dibujar sobre el video (Overlay)
    const drawResults = (data) => {
        const canvas = canvasRef.current
        if (!canvas || !videoRef.current) return

        const ctx = canvas.getContext('2d')
        canvas.width = videoRef.current.videoWidth
        canvas.height = videoRef.current.videoHeight

        // Limpiar previo
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Si hay datos de manos
        const hands = data.hands || []
        if (hands.length === 0) return

        // Dibujar esqueleto
        hands.forEach(hand => {
            const landmarks = hand.landmarks

            // Dibujar puntos
            ctx.fillStyle = '#00FF00'
            landmarks.forEach(point => {
                const x = point[0] * canvas.width
                const y = point[1] * canvas.height
                ctx.beginPath()
                ctx.arc(x, y, 4, 0, 2 * Math.PI)
                ctx.fill()
            })

            // Dibujar conexiones
            ctx.strokeStyle = '#00FF00'
            ctx.lineWidth = 2
            drawConnection(ctx, landmarks, [0, 1, 2, 3, 4], canvas.width, canvas.height)
            drawConnection(ctx, landmarks, [0, 5, 6, 7, 8], canvas.width, canvas.height)
            drawConnection(ctx, landmarks, [5, 9, 10, 11, 12], canvas.width, canvas.height)
            drawConnection(ctx, landmarks, [9, 13, 14, 15, 16], canvas.width, canvas.height)
            drawConnection(ctx, landmarks, [13, 17, 18, 19, 20], canvas.width, canvas.height)
            drawConnection(ctx, landmarks, [0, 17], canvas.width, canvas.height)
        })
    }

    const drawConnection = (ctx, landmarks, indices, w, h) => {
        ctx.beginPath()
        ctx.moveTo(landmarks[indices[0]][0] * w, landmarks[indices[0]][1] * h)
        for (let i = 1; i < indices.length; i++) {
            ctx.lineTo(landmarks[indices[i]][0] * w, landmarks[indices[i]][1] * h)
        }
        ctx.stroke()
    }

    // Lógica para traducir la secuencia completa a lenguaje natural
    const synthesizeSentence = async () => {
        if (bufferRef.current.length === 0 || isProcessingBuffer) return

        setIsProcessingBuffer(true)
        setAiContext("✨ IA procesando frase...")
        setNaturalTranslation("") // Limpiar anterior

        try {
            console.log("Enviando secuencia para traducción:", bufferRef.current);
            const res = await api.post('/recognize/translate-sequence', {
                gestures: bufferRef.current
            })

            if (res.data.success) {
                const { translation, method } = res.data
                setNaturalTranslation(translation)

                if (method === 'llm') {
                    setAiContext("¡Frase traducida con éxito!")
                } else {
                    setAiContext("Generando frase básica (IA no disponible)")
                }

                speak(translation)

                // Limpiar buffer tras éxito
                bufferRef.current = []
                setGestureBuffer([])
                lastSpokenGesture.current = null
            } else {
                setAiContext("No se pudo completar la traducción.")
            }
        } catch (err) {
            console.error("Error en translate-sequence:", err)
            // Fallback LOCAL extremo: simplemente unir palabras
            const words = bufferRef.current.join(" ")
            const fallback = words.charAt(0).toUpperCase() + words.slice(1) + "."
            setNaturalTranslation(fallback)
            setAiContext("Conexión perdida (usando traducción local)")
            speak(fallback)

            // Limpiar de todos modos para permitir nueva frase
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
        setAiContext("Listo para nueva oración")
    }

    // Lógica de Razonamiento IA
    const handleAiReasoning = (data) => {
        // 1. Lógica para Modo Traducción
        if (mode === 'traduccion') {
            const gesture = data.gesture
            const confidence = data.confidence || 0

            // 1.1 Filtro de ruido: ignoramos cualquier cosa con confianza menor a 0.4
            if (!gesture || confidence < 0.4) {
                stabilityCounter.current = 0
                currentCandidate.current = null
                silenceCounter.current += 1

                // Reset tras pausa corta
                if (silenceCounter.current > 10) {
                    lastSpokenGesture.current = null
                }

                // Traducción automática tras ~3 segundos (30 frames x 100ms)
                if (silenceCounter.current === 30) {
                    if (bufferRef.current.length > 0) {
                        synthesizeSentence()
                    } else {
                        setAiContext("Esperando señas...")
                    }
                }
                return
            }

            // 1.2 Si entramos aquí, hay algo detectable
            silenceCounter.current = 0

            // 1.3 Debounce: contar frames consecutivos
            if (gesture === currentCandidate.current) {
                stabilityCounter.current += 1
            } else {
                currentCandidate.current = gesture
                stabilityCounter.current = 1
            }

            // 1.4 Umbrales de captura Estrictos para evitar "falsos positivos"
            // Si la confianza es media (0.4 - 0.7), te avisamos pero NO guardamos
            if (confidence < 0.7) {
                setAiContext(`IA cree ver: ${gesture} (Sé más claro)`)
                return
            }

            // Captura definitiva: requiere 4 frames estables (~400ms) y alta confianza (> 0.7)
            if (stabilityCounter.current >= 4) {
                if (lastSpokenGesture.current !== gesture) {
                    lastSpokenGesture.current = gesture
                    setLastPrediction(gesture)

                    // AÑADIR AL BUFFER REAL (aquí es donde se "guarda")
                    bufferRef.current = [...bufferRef.current, gesture]
                    setGestureBuffer([...bufferRef.current])
                    setAiContext(`✅ Capturado: ${gesture}`)
                }
            }
        }
        // 2. Lógica para Modo Exploración
        else if (mode === 'exploracion') {
            const numHands = data.hands_detected || 0
            if (numHands > 0 && lastPrediction !== 'hands_detected') {
                setLastPrediction('hands_detected')
                setAiContext(numHands === 1 ? "Veo una mano." : "Veo dos manos.")
            } else if (numHands === 0 && lastPrediction !== 'none') {
                setLastPrediction('none')
                setAiContext("Esperando detección...")
            }
        }
    }

    return (
        <div className="traductor-container">
            {/* Área Principal de Video */}
            <div className="video-wrapper">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="main-video"
                />
                <canvas
                    ref={canvasRef}
                    className="overlay-canvas"
                />

                {/* Overlay de UI */}
                <div className="ui-overlay">
                    <div className="status-badge">
                        <span className={`dot ${isActive ? 'active' : ''}`}></span>
                        {mode === 'traduccion' ? ' Modo Traductor IA (Natural)' : ' Modo Exploración'}
                    </div>

                    {/* Panel de IA Mejorado */}
                    <div className="ai-panel glass-effect">
                        <div className="ai-header">
                            <span className="ai-icon">✨</span>
                            <span className="ai-title">Traductor Inteligente</span>
                        </div>

                        <div className="ai-content">
                            {mode === 'traduccion' ? (
                                <>
                                    <div className="gesture-sequence">
                                        <label>Secuencia capturada:</label>
                                        <div className="buffer-tags">
                                            {gestureBuffer.length > 0 ? (
                                                gestureBuffer.map((g, i) => (
                                                    <span key={i} className="gesture-tag pulse">{g}</span>
                                                ))
                                            ) : (
                                                <span className="placeholder">Haz varias señas para formar una frase...</span>
                                            )}
                                        </div>
                                    </div>

                                    {naturalTranslation && (
                                        <div className="final-translation highlight">
                                            <label>Traducción natural:</label>
                                            <p>{naturalTranslation}</p>
                                        </div>
                                    )}

                                    <div className="live-status">
                                        <small className={isProcessingBuffer ? "loading" : ""}>
                                            {isProcessingBuffer ? "IA pensando..." : aiContext}
                                        </small>
                                    </div>
                                </>
                            ) : (
                                <p>{aiContext}</p>
                            )}
                        </div>

                        {isSpeaking && <div className="speaking-indicator">🔊 Hablando...</div>}
                    </div>

                    {/* Controles Flotantes */}
                    <div className="controls glass-effect">
                        <button
                            onClick={() => {
                                setMode(m => m === 'exploracion' ? 'traduccion' : 'exploracion')
                                resetBuffer()
                            }}
                            className="btn-toggle"
                        >
                            {mode === 'exploracion' ? 'Activar Traducción' : 'Vista Simple'}
                        </button>

                        {mode === 'traduccion' && (
                            <>
                                <button
                                    onClick={synthesizeSentence}
                                    className="btn-action"
                                    disabled={gestureBuffer.length === 0 || isProcessingBuffer}
                                    title="Traducir buffer ahora"
                                >
                                    ✨ Traducir
                                </button>
                                <button
                                    onClick={resetBuffer}
                                    className="btn-icon"
                                    title="Borrar buffer"
                                >
                                    🗑️
                                </button>
                            </>
                        )}

                        <button onClick={() => speak("Prueba de audio.")} className="btn-icon" title="Prueba de voz">
                            🔊
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
