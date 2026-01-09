import React, { useState, useEffect, useCallback } from 'react'
import api from '../services/api'
import './VerSenasOrganizado.css'

export default function VerSenasOrganizado() {
    const [gestures, setGestures] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [activeCategory, setActiveCategory] = useState('all')
    const [selectedGesture, setSelectedGesture] = useState(null)
    const [isAddingMode, setIsAddingMode] = useState(false)
    const [addingSamples, setAddingSamples] = useState(null)

    // Categorías predefinidas
    const categories = [
        { id: 'all', label: '📋 Todas', icon: '📋' },
        { id: 'numbers', label: '🔢 Números', icon: '🔢' },
        { id: 'alphabet', label: '🔤 Abecedario', icon: '🔤' },
        { id: 'greetings', label: '👋 Saludos', icon: '👋' },
        { id: 'common', label: '💬 Comunes', icon: '💬' },
        { id: 'other', label: '📦 Otras', icon: '📦' }
    ]

    // Patrones para clasificación automática con IA
    const classifyGesture = useCallback((name) => {
        const normalizedName = name.toLowerCase().trim()

        // Números
        const numberPatterns = [
            /^(cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)$/,
            /^[0-9]+$/,
            /^(once|doce|trece|catorce|quince|veinte|treinta|cuarenta|cincuenta|cien|mil)$/,
            /^numero_?\d+$/i
        ]

        // Abecedario
        const alphabetPatterns = [
            /^[a-z]$/,
            /^letra_?[a-z]$/i,
            /^(a|b|c|d|e|f|g|h|i|j|k|l|m|n|ñ|o|p|q|r|s|t|u|v|w|x|y|z)$/i
        ]

        // Saludos
        const greetingPatterns = [
            /^(hola|adios|buenos_?dias|buenas_?tardes|buenas_?noches|gracias|por_?favor|permiso|disculpa|hasta_?luego|bienvenido|saludos?)$/i
        ]

        // Palabras comunes
        const commonPatterns = [
            /^(si|no|quiero|necesito|ayuda|agua|comida|baño|doctor|emergencia|familia|amigo|casa|trabajo|escuela|dinero|tiempo|hoy|mañana|ayer)$/i
        ]

        // Verificar cada categoría
        for (const pattern of numberPatterns) {
            if (pattern.test(normalizedName)) return 'numbers'
        }

        for (const pattern of alphabetPatterns) {
            if (pattern.test(normalizedName)) return 'alphabet'
        }

        for (const pattern of greetingPatterns) {
            if (pattern.test(normalizedName)) return 'greetings'
        }

        for (const pattern of commonPatterns) {
            if (pattern.test(normalizedName)) return 'common'
        }

        return 'other'
    }, [])

    const fetchGestures = async () => {
        try {
            setLoading(true)
            const res = await api.get('/gestures/list')
            const rawGestures = res.data.gestures || []

            // Clasificar automáticamente cada gesto
            const classifiedGestures = rawGestures.map(g => ({
                ...g,
                category: classifyGesture(g.name),
                displayName: formatDisplayName(g.name)
            }))

            // Ordenar alfabéticamente dentro de cada categoría
            classifiedGestures.sort((a, b) => a.displayName.localeCompare(b.displayName))

            setGestures(classifiedGestures)
            setLoading(false)
        } catch (err) {
            console.error(err)
            setError('No se pudieron cargar las señas guardadas.')
            setLoading(false)
        }
    }

    const formatDisplayName = (name) => {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase())
    }

    useEffect(() => {
        fetchGestures()
    }, [])

    const handleDelete = async (name) => {
        if (!window.confirm(`¿Estás seguro de eliminar la seña "${name}" y todas sus fotos?`)) return

        try {
            await api.delete(`/gestures/${name}`)
            fetchGestures()
        } catch (err) {
            alert('Error al eliminar')
        }
    }

    const handleCorrect = (gesture) => {
        setSelectedGesture(gesture)
        // Aquí se podría abrir un modal para corregir/renombrar
        const newName = window.prompt(`Corregir nombre de "${gesture.name}":`, gesture.name)
        if (newName && newName !== gesture.name) {
            // Implementar lógica de renombrar
            alert(`Renombrar de "${gesture.name}" a "${newName}" - Funcionalidad próximamente`)
        }
        setSelectedGesture(null)
    }

    const startAddingSamples = async (gesture) => {
        setAddingSamples(gesture.name)
        // Redirigir o abrir modal para agregar muestras
        // Por ahora mostramos un mensaje
        setTimeout(() => {
            alert(`Para agregar muestras a "${gesture.name}", ve a "Grabar Seña" y selecciona esta seña.`)
            setAddingSamples(null)
        }, 500)
    }

    const getFilteredGestures = () => {
        if (activeCategory === 'all') return gestures
        return gestures.filter(g => g.category === activeCategory)
    }

    const getCategoryCount = (categoryId) => {
        if (categoryId === 'all') return gestures.length
        return gestures.filter(g => g.category === categoryId).length
    }

    const getSampleStatus = (samples) => {
        if (samples >= 50) return { class: 'excellent', label: 'Excelente', icon: '⭐' }
        if (samples >= 30) return { class: 'good', label: 'Suficiente', icon: '✅' }
        if (samples >= 15) return { class: 'warning', label: 'Mínimo', icon: '⚠️' }
        return { class: 'danger', label: 'Insuficiente', icon: '❌' }
    }

    const getCategoryIcon = (category) => {
        const cat = categories.find(c => c.id === category)
        return cat ? cat.icon : '📦'
    }

    if (loading) {
        return (
            <div className="ver-senas-loading">
                <div className="loading-spinner"></div>
                <p>Cargando señas guardadas...</p>
            </div>
        )
    }

    return (
        <div className="ver-senas-organizado">
            {/* Header */}
            <header className="senas-header">
                <div className="header-content">
                    <h1>📂 Biblioteca de Señas</h1>
                    <p>Señas organizadas automáticamente por categoría • {gestures.length} señas registradas</p>
                </div>
                <div className="header-stats">
                    <div className="stat-badge">
                        <span className="stat-value">{gestures.reduce((s, g) => s + (g.samples || 0), 0)}</span>
                        <span className="stat-label">muestras totales</span>
                    </div>
                </div>
            </header>

            {/* Categorías */}
            <nav className="categories-nav">
                {categories.map(cat => (
                    <button
                        key={cat.id}
                        className={`category-btn ${activeCategory === cat.id ? 'active' : ''}`}
                        onClick={() => setActiveCategory(cat.id)}
                    >
                        <span className="cat-icon">{cat.icon}</span>
                        <span className="cat-label">{cat.label}</span>
                        <span className="cat-count">{getCategoryCount(cat.id)}</span>
                    </button>
                ))}
            </nav>

            {error && <p className="error-message">{error}</p>}

            {/* Grid de Señas */}
            <section className="gestures-section">
                {getFilteredGestures().length === 0 ? (
                    <div className="empty-state">
                        <span className="empty-icon">📭</span>
                        <h3>No hay señas en esta categoría</h3>
                        <p>Ve a "Grabar Seña" para empezar a agregar nuevas señas.</p>
                    </div>
                ) : (
                    <div className="gestures-grid">
                        {getFilteredGestures().map((g) => {
                            const sampleStatus = getSampleStatus(g.samples)
                            return (
                                <div
                                    key={g.name}
                                    className={`gesture-card ${addingSamples === g.name ? 'adding' : ''}`}
                                >
                                    {/* Encabezado de la tarjeta */}
                                    <div className="card-header">
                                        <span className="category-badge">
                                            {getCategoryIcon(g.category)}
                                        </span>
                                        <div className="card-actions">
                                            <button
                                                className="action-btn edit"
                                                onClick={() => handleCorrect(g)}
                                                title="Corregir nombre"
                                            >
                                                ✏️
                                            </button>
                                            <button
                                                className="action-btn delete"
                                                onClick={() => handleDelete(g.name)}
                                                title="Eliminar"
                                            >
                                                🗑️
                                            </button>
                                        </div>
                                    </div>

                                    {/* Icono/Preview */}
                                    <div className="card-preview">
                                        <span className="preview-icon">✋</span>
                                    </div>

                                    {/* Información */}
                                    <div className="card-info">
                                        <h3 className="gesture-name">{g.displayName}</h3>

                                        {/* Muestras con indicador de estado */}
                                        <div className={`samples-indicator ${sampleStatus.class}`}>
                                            <span className="samples-icon">{sampleStatus.icon}</span>
                                            <span className="samples-count">{g.samples} muestras</span>
                                            <span className="samples-status">{sampleStatus.label}</span>
                                        </div>

                                        {/* Barra de progreso */}
                                        <div className="samples-progress">
                                            <div
                                                className={`progress-fill ${sampleStatus.class}`}
                                                style={{ width: `${Math.min((g.samples / 50) * 100, 100)}%` }}
                                            ></div>
                                        </div>

                                        {/* Botón para agregar muestras */}
                                        <button
                                            className="btn-add-samples"
                                            onClick={() => startAddingSamples(g)}
                                            disabled={addingSamples === g.name}
                                        >
                                            {addingSamples === g.name ? (
                                                <>
                                                    <span className="btn-spinner"></span>
                                                    Preparando...
                                                </>
                                            ) : (
                                                <>
                                                    ➕ Agregar Muestras
                                                </>
                                            )}
                                        </button>

                                        {/* Recomendación */}
                                        {g.samples < 30 && (
                                            <p className="recommendation">
                                                💡 Se recomiendan al menos 30 muestras para un reconocimiento preciso
                                            </p>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                )}
            </section>

            {/* Información de categorías */}
            <section className="category-info">
                <h3>📊 Resumen por Categoría</h3>
                <div className="category-summary">
                    {categories.filter(c => c.id !== 'all').map(cat => {
                        const count = getCategoryCount(cat.id)
                        const samples = gestures
                            .filter(g => g.category === cat.id)
                            .reduce((s, g) => s + (g.samples || 0), 0)

                        return (
                            <div key={cat.id} className="summary-card">
                                <span className="summary-icon">{cat.icon}</span>
                                <div className="summary-info">
                                    <span className="summary-label">{cat.label}</span>
                                    <span className="summary-stats">
                                        {count} señas • {samples} muestras
                                    </span>
                                </div>
                                <div className="summary-bar">
                                    <div
                                        className="bar-fill"
                                        style={{ width: `${gestures.length > 0 ? (count / gestures.length) * 100 : 0}%` }}
                                    ></div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </section>

            {/* Leyenda */}
            <section className="legend-section">
                <h4>Guía de Estados</h4>
                <div className="legend-items">
                    <div className="legend-item">
                        <span className="legend-badge excellent">⭐ Excelente</span>
                        <span>50+ muestras</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-badge good">✅ Suficiente</span>
                        <span>30-49 muestras</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-badge warning">⚠️ Mínimo</span>
                        <span>15-29 muestras</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-badge danger">❌ Insuficiente</span>
                        <span>&lt;15 muestras</span>
                    </div>
                </div>
            </section>
        </div>
    )
}
