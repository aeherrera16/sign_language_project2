import React, { useEffect, useState } from 'react'

const slides = [
  {
    title: '¿Qué es la inclusión?',
    img: '/src/assets/carrusel_quees.jpg',
    content: [
      'La inclusión es la incorporación como iguales en la sociedad o una organización de individuos de diferentes grupos.'
    ]
  },
  {
    title: 'Tipos de inclusión',
    img: '/src/assets/carrusel_tipos.jpg',
    content: [
      'Social', 'Educativa', 'Laboral', 'Digital'
    ]
  },
  {
    title: 'Características',
    img: '/src/assets/carrusel_caracteristicas.jpg',
    content: [
      'Respeto', 'Accesibilidad', 'Equidad', 'Participación'
    ]
  },
  {
    title: 'Ejemplos',
    img: '/src/assets/carrusel_ejemplos.jpg',
    content: [
      'Programas de accesibilidad', 'Intérpretes de lengua de señas', 'Tecnologías asistivas', 'Políticas inclusivas'
    ]
  }
]

export default function InfoCarousel() {
  const [idx, setIdx] = useState(0)
  useEffect(() => {
    const timer = setTimeout(() => setIdx((idx+1)%slides.length), 5000)
    return () => clearTimeout(timer)
  }, [idx])

  const goTo = i => setIdx(i)
  const prev = () => setIdx(idx === 0 ? slides.length-1 : idx-1)
  const next = () => setIdx((idx+1)%slides.length)

  return (
    <div className="carousel-card">
      <div className="carousel-imgbox">
        <img src={slides[idx].img} alt={slides[idx].title} />
      </div>
      <div className="carousel-content">
        <h3>{slides[idx].title}</h3>
        <ul>
          {slides[idx].content.map((c,i) => <li key={i}>{c}</li>)}
        </ul>
        <div className="carousel-controls">
          <button className="carousel-arrow" onClick={prev} aria-label="Anterior">&#8592;</button>
          <div className="carousel-dots">
            {slides.map((_,i) => (
              <span key={i} className={i===idx?"dot active":"dot"} onClick={()=>goTo(i)}></span>
            ))}
          </div>
          <button className="carousel-arrow" onClick={next} aria-label="Siguiente">&#8594;</button>
        </div>
      </div>
    </div>
  )
}
