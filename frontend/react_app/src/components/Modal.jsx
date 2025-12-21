import React from 'react'

export default function Modal({ title, message, onClose, type = 'info' }) {
    // type: 'info', 'success', 'error'

    const colors = {
        info: '#3b82f6',
        success: '#10b981',
        error: '#ef4444'
    }

    const icons = {
        info: 'ℹ️',
        success: '✅',
        error: '⚠️'
    }

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(5px)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
            animation: 'fadeIn 0.2s'
        }} onClick={onClose}>
            <div style={{
                background: 'rgba(255, 255, 255, 0.95)',
                padding: '30px',
                borderRadius: '20px',
                width: '90%',
                maxWidth: '400px',
                textAlign: 'center',
                boxShadow: '0 20px 50px rgba(0,0,0,0.3)',
                transform: 'translateY(0)',
                animation: 'slideUp 0.3s',
                borderTop: `6px solid ${colors[type]}`
            }} onClick={e => e.stopPropagation()}>

                <div style={{ fontSize: '3em', marginBottom: '10px' }}>
                    {icons[type]}
                </div>

                <h3 style={{ color: '#1f2937', marginBottom: '15px', fontSize: '1.5em' }}>
                    {title}
                </h3>

                <p style={{ color: '#4b5563', lineHeight: '1.6', marginBottom: '25px', fontSize: '1.1em' }}>
                    {message}
                </p>

                <button onClick={onClose} style={{
                    background: colors[type],
                    color: 'white',
                    border: 'none',
                    padding: '12px 30px',
                    borderRadius: '10px',
                    fontSize: '1em',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    transition: 'transform 0.1s',
                    width: '100%'
                }}
                    onMouseOver={e => e.target.style.transform = 'scale(1.02)'}
                    onMouseOut={e => e.target.style.transform = 'scale(1)'}
                >
                    Entendido
                </button>
            </div>
            <style>
                {`
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
                `}
            </style>
        </div>
    )
}
