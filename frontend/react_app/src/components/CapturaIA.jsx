import React, { useRef, useEffect, useState } from 'react'
import api from '../services/api'
import '../styles/admin.css'

export default function CapturaIA(){
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const maskCanvasRef = useRef(null)
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [gestureName, setGestureName] = useState('')
  const [overlayUrl, setOverlayUrl] = useState(null)
  const [saving, setSaving] = useState(false)
  const qualityThreshold = 70 // default min score to allow save

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if(videoRef.current){
          videoRef.current.srcObject = stream
        }
      }).catch(err => console.error('cámara no disponible', err))
  }, [])

  useEffect(() => {
    if(!analyzing) return
    const id = setInterval(() => {
      analyzeFrame()
    }, 2000)
    return () => clearInterval(id)
  }, [analyzing])

  const preprocessAndDraw = (video, canvas) => {
    // simple client-side preprocessing to help low-res/poor lighting
    const ctx = canvas.getContext('2d')
    // apply CSS-like filters for contrast/saturation boost
    ctx.filter = 'contrast(130%) saturate(120%)'
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    ctx.filter = 'none'
  }

  const analyzeFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if(!video || !canvas) return
    // set canvas size depending on video; upscale if very small
    const targetW = Math.max(480, video.videoWidth)
    const targetH = Math.max(360, video.videoHeight)
    canvas.width = targetW
    canvas.height = targetH

    // preprocess and draw
    preprocessAndDraw(video, canvas)

    // get image blob
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9))
    if(!blob) return

    // call segmentation endpoint to get overlay image
    try{
      const fd = new FormData()
      fd.append('file', blob, 'frame.jpg')
      const segRes = await fetch(api.defaults.baseURL.replace('/api','') + '/api/capture/segment-frame', {
        method: 'POST',
        body: fd
      })

      if(segRes.ok){
        const segBlob = await segRes.blob()
        const url = URL.createObjectURL(segBlob)
        setOverlayUrl(url)
        // draw overlay into mask canvas if present
        const maskCanvas = maskCanvasRef.current
        if(maskCanvas){
          maskCanvas.width = targetW
          maskCanvas.height = targetH
          const mctx = maskCanvas.getContext('2d')
          const img = new Image()
          img.onload = ()=>{
            mctx.clearRect(0,0,maskCanvas.width,maskCanvas.height)
            mctx.globalAlpha = 0.8
            mctx.drawImage(img,0,0,maskCanvas.width,maskCanvas.height)
          }
          img.src = url
        }
      }
    }catch(e){
      console.warn('segmentación falló', e)
    }

    // send for analysis (base64) using smaller payload (dataURL)
    const base64 = canvas.toDataURL('image/jpeg', 0.8)

    try{
      const res = await api.post('/capture/analyze-capture', { image_base64: base64, gesture_name: gestureName })
      setAnalysis(res.data)
    }catch(e){
      console.error('error analizando', e)
    }
  }

  const handleCapture = async () => {
    await analyzeFrame()
    if(!analysis){
      alert('No se pudo analizar la captura')
      return
    }

    if(analysis.is_good && analysis.score >= qualityThreshold){
      // prepare blob from canvas
      const canvas = canvasRef.current
      canvas.toBlob(async (blob)=>{
        if(!blob) return
        setSaving(true)
        try{
          const form = new FormData()
          form.append('image', blob, `${gestureName || 'unknown'}.jpg`)
          // we use gestureName as metadata; ideally use gesture_id
          form.append('gesture_id', '')
          form.append('metadata', JSON.stringify({ gesture_name: gestureName, analysis }))

          const saveRes = await api.post('/capture/save', form, { headers: { 'Content-Type': 'multipart/form-data' }})
          // Some servers expect /api/capture/save or /api/capture/save - our backend router is /api/capture/save
          if(saveRes.status === 200 || saveRes.status === 201){
            alert('Captura guardada con ID: ' + (saveRes.data.id || 'OK'))
          }else{
            alert('Error al guardar captura')
          }
        }catch(e){
          console.error('error guardando', e)
          alert('Error guardando la captura')
        }finally{
          setSaving(false)
        }
      }, 'image/jpeg', 0.9)
    }else{
      alert('Captura no cumple umbral de calidad: ' + (analysis ? analysis.score : 'N/A'))
    }
  }

  return (
    <div className="captura-ia-view">
      <div className="video-section" style={{position:'relative'}}>
        <video ref={videoRef} autoPlay playsInline muted style={{width:'100%',borderRadius:8}} />
        <canvas ref={canvasRef} style={{display:'none'}} />
        <canvas ref={maskCanvasRef} style={{position:'absolute',top:0,left:0,right:0,bottom:0,width:'100%',height:'100%',pointerEvents:'none',borderRadius:8}} />
      </div>

      <div className="control-panel" style={{marginTop:16}}>
        <label>Nombre de la seña</label>
        <input value={gestureName} onChange={(e)=>setGestureName(e.target.value)} placeholder="Ej: HOLA" />

        <div style={{marginTop:10,display:'flex',gap:8}}>
          <button className="btn primary" onClick={()=>setAnalyzing(!analyzing)}>{analyzing ? '⏸️ Pausar' : '▶️ Iniciar'}</button>
          <button className="btn outline" onClick={handleCapture}>📸 Capturar</button>
        </div>

        {analysis && (
          <div style={{marginTop:12}} className={`analysis-panel quality-${analysis.quality}`}>
            <h4>Calidad: {analysis.quality} ({analysis.score}/100)</h4>
            <div>Manos: {analysis.metrics.hands_percentage.toFixed(1)}% - Detectadas: {analysis.metrics.hands_detected}</div>
            <ul>
              {analysis.recommendations.map((r,i)=> <li key={i}>{r}</li>)}
            </ul>
            <div style={{marginTop:8,display:'flex',gap:8}}>
              <button className="btn primary" onClick={handleCapture} disabled={saving}>{saving ? 'Guardando...' : '💾 Guardar Seña'}</button>
              <button className="btn outline" onClick={()=>{ if(overlayUrl) window.open(overlayUrl,'_blank') }}>🔍 Ver Segmentación</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
