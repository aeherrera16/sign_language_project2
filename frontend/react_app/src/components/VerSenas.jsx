import React, { useState, useEffect } from 'react'
import api from '../services/api'

export default function VerSenas() {
    const [gestures, setGestures] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')

    const fetchGestures = async () => {
        try {
            const res = await api.get('/gestures/list')
            setGestures(res.data.gestures || [])
            setLoading(false)
        } catch (err) {
            console.error(err)
            setError('No se pudieron cargar las señas guardadas.')
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchGestures()
    }, [])

    const handleDelete = async (name) => {
        if (!window.confirm(`¿Estás seguro de eliminar la seña "${name}" y todas sus fotos?`)) return

        try {
            await api.delete(`/gestures/${name}`)
            fetchGestures() // Recargar lista
        } catch (err) {
            alert('Error al eliminar')
        }
    }

    return (
        <div className="admin-content">
            <div className="container">
                <h2>📂 Señas Guardadas</h2>
                <p>Aquí puedes ver el "cerebro" de tu IA: las carpetas con las fotos que has grabado.</p>

                {loading && <p>Cargando...</p>}
                {error && <p style={{ color: 'red' }}>{error}</p>}

                {!loading && gestures.length === 0 && (
                    <div className="empty-state" style={{ textAlign: 'center', padding: '40px', background: '#f8f9fa', borderRadius: '10px' }}>
                        <h3>📭 No hay señas todavía</h3>
                        <p>Ve a "Grabar Seña" para empezar.</p>
                    </div>
                )}

                <div className="gestures-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px', marginTop: '20px' }}>
                    {gestures.map((g) => (
                        <div key={g.name} className="gesture-card" style={{ border: '1px solid #ddd', borderRadius: '12px', overflow: 'hidden', background: 'white', boxShadow: '0 2px 5px rgba(0,0,0,0.05)' }}>
                            <div style={{ height: '150px', background: '#e9ecef', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '3em' }}>
                                ✋
                            </div>
                            <div style={{ padding: '15px' }}>
                                <h3 style={{ margin: '0 0 5px 0', textTransform: 'capitalize' }}>{g.name}</h3>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{
                                        background: g.samples >= 30 ? '#d1e7dd' : '#fff3cd',
                                        color: g.samples >= 30 ? '#0f5132' : '#856404',
                                        padding: '4px 8px', borderRadius: '4px', fontSize: '0.9em', fontWeight: 'bold'
                                    }}>
                                        {g.samples} muestras
                                    </span>
                                    <button
                                        onClick={() => handleDelete(g.name)}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.2em' }}
                                        title="Eliminar"
                                    >
                                        🗑️
                                    </button>
                                </div>
                                {g.samples < 30 && (
                                    <small style={{ display: 'block', marginTop: '10px', color: '#dc3545' }}>
                                        ⚠️ Se recomiendan +30
                                    </small>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
