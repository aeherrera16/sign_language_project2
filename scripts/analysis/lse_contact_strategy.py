# -*- coding: utf-8 -*-
"""
🇪🇨 GUÍA DE CONTACTOS Y ESTRATEGIAS LSE ECUADOR
Información específica para ejecutar el plan de mejora del dataset
"""

import json
import os
from datetime import datetime

class LSEContactStrategy:
    def __init__(self):
        self.contact_database = {}
        
    def create_fenasec_approach(self):
        """Estrategia específica para FENASEC"""
        fenasec_strategy = {
            "organizacion": "FENASEC - Federación Nacional de Sordos del Ecuador",
            "importancia": "CRÍTICA - Organización oficial más importante",
            "contactos_principales": {
                "presidente": {
                    "nombre": "Por confirmar en sitio web oficial",
                    "rol": "Presidente Nacional FENASEC",
                    "contacto_recomendado": "A través de página web oficial"
                },
                "secretaria_ejecutiva": {
                    "rol": "Coordinación ejecutiva",
                    "ubicacion": "Quito - Sede principal"
                }
            },
            "canales_contacto": {
                "sitio_web": "http://www.fenasec.org.ec",
                "redes_sociales": [
                    "Facebook: FENASEC Ecuador",
                    "Instagram: @fenasec_ecuador"
                ],
                "presencial": "Visita a sede en Quito (recomendado para primera reunión)"
            },
            "propuesta_inicial": {
                "documento": "Carta formal de presentación",
                "contenido_clave": [
                    "Presentación del proyecto LSE Ecuador",
                    "Beneficios específicos para la comunidad sorda",
                    "Solicitud de reunión para presentación detallada",
                    "Ofrecimiento de tecnología gratuita para FENASEC"
                ],
                "adjuntos": [
                    "Resumen ejecutivo del proyecto",
                    "Demostración del sistema actual (video)",
                    "Propuesta de colaboración específica"
                ]
            },
            "estrategia_reunión": {
                "ubicacion_sugerida": "Sede FENASEC en Quito",
                "participantes_necesarios": [
                    "Presidente o Vicepresidente FENASEC",
                    "Coordinador técnico FENASEC",
                    "Intérpretes seniors de confianza",
                    "Director del proyecto LSE"
                ],
                "agenda_propuesta": [
                    "Presentación del proyecto actual (20 min)",
                    "Demostración en vivo del sistema (15 min)",
                    "Explicación de beneficios para comunidad (15 min)",
                    "Propuesta de colaboración específica (20 min)",
                    "Discusión y preguntas (30 min)",
                    "Definición de próximos pasos (10 min)"
                ]
            },
            "beneficios_a_destacar": [
                "🎯 Tecnología de reconocimiento LSE completamente GRATUITA para FENASEC",
                "📱 Aplicación móvil personalizada con logo FENASEC",
                "🎓 Capacitación completa para miembros de FENASEC",
                "🏆 Reconocimiento internacional como pioneros en tecnología LSE",
                "💡 Participación en publicaciones académicas internacionales",
                "🌟 Posicionamiento de Ecuador como líder regional en IA inclusiva"
            ]
        }
        return fenasec_strategy
    
    def create_university_contacts(self):
        """Contactos específicos de universidades"""
        universities = {
            "universidad_tecnica_ambato": {
                "nombre": "Universidad Técnica de Ambato",
                "importancia": "ALTA - Base del proyecto actual",
                "contactos": {
                    "biblioteca": {
                        "email": "biblioteca@uta.edu.ec",
                        "telefono": "593-3-2848487",
                        "responsable": "Coordinación de biblioteca"
                    },
                    "investigacion": {
                        "departamento": "Dirección de Investigación y Desarrollo",
                        "contacto": "A través del sitio web institucional"
                    },
                    "facultad_sistemas": {
                        "nombre": "Facultad de Ingeniería en Sistemas",
                        "relevancia": "Área técnica directamente relacionada"
                    }
                },
                "estrategia": [
                    "Solicitar reunión con Dirección de Investigación",
                    "Presentar como proyecto bandera de la universidad",
                    "Proponer colaboración con estudiantes de tesis",
                    "Buscar financiamiento interno para investigación"
                ]
            },
            "universidad_central": {
                "nombre": "Universidad Central del Ecuador",
                "ubicacion": "Quito",
                "contactos": {
                    "investigacion": {
                        "direccion": "Dirección General de Investigación",
                        "relevancia": "Financiamiento y recursos"
                    },
                    "sistemas": {
                        "facultad": "Facultad de Ingeniería y Ciencias Aplicadas",
                        "carrera": "Ingeniería en Sistemas"
                    }
                }
            },
            "escuela_politecnica_nacional": {
                "nombre": "Escuela Politécnica Nacional",
                "fortaleza": "Excelencia técnica e investigación",
                "contactos": {
                    "sistemas": {
                        "departamento": "Departamento de Informática y Ciencias de la Computación",
                        "relevancia": "IA y Machine Learning"
                    },
                    "investigacion": {
                        "oficina": "Vicerrectorado de Investigación y Proyección Social"
                    }
                }
            },
            "universidad_san_francisco": {
                "nombre": "Universidad San Francisco de Quito",
                "fortaleza": "Recursos internacionales y tecnología",
                "contactos": {
                    "sistemas": {
                        "colegio": "Colegio de Ciencias e Ingenierías",
                        "relevancia": "Recursos técnicos avanzados"
                    }
                }
            }
        }
        return universities
    
    def create_government_contacts(self):
        """Contactos gubernamentales específicos"""
        government = {
            "conadis": {
                "nombre": "Consejo Nacional para la Igualdad de Discapacidades",
                "importancia": "CRÍTICA - Organismo rector",
                "contactos": {
                    "principal": {
                        "telefono": "593-2-2459243",
                        "extension": "Solicitar extensión apropiada",
                        "horario": "Lunes a Viernes 8:00-17:00"
                    },
                    "direccion_tecnica": {
                        "responsable": "Dirección Técnica de Accesibilidad",
                        "relevancia": "Área directamente relacionada con tecnología"
                    }
                },
                "estrategia_contacto": [
                    "Llamada inicial para solicitar reunión",
                    "Envío de propuesta formal por correo",
                    "Seguimiento presencial en oficinas de Quito",
                    "Presentación como proyecto de política pública"
                ],
                "propuesta_valor": [
                    "Cumplimiento de objetivos CONADIS",
                    "Tecnología inclusiva nacional",
                    "Referente internacional para Ecuador",
                    "Implementación en servicios públicos"
                ]
            },
            "mies": {
                "nombre": "Ministerio de Inclusión Económica y Social",
                "relevancia": "Financiamiento y política social",
                "contactos": {
                    "inclusion": {
                        "subsecretaria": "Subsecretaría de Discapacidades",
                        "relevancia": "Políticas específicas"
                    }
                }
            },
            "senescyt": {
                "nombre": "Secretaría de Educación Superior, Ciencia, Tecnología e Innovación",
                "relevancia": "Financiamiento de investigación",
                "programas_relevantes": [
                    "Proyectos de Investigación de Desarrollo Tecnológico",
                    "Fondo Nacional de Ciencia y Tecnología",
                    "Programas de Innovación Social"
                ]
            }
        }
        return government
    
    def create_international_contacts(self):
        """Contactos internacionales potenciales"""
        international = {
            "organismos_internacionales": {
                "unesco": {
                    "programa": "UNESCO - Tecnología para la Inclusión",
                    "relevancia": "Financiamiento internacional"
                },
                "bid": {
                    "programa": "BID Lab - Innovación Social",
                    "relevancia": "Fondos para proyectos tecnológicos inclusivos"
                },
                "onu_discapacidad": {
                    "programa": "ONU - Convención sobre Derechos de Personas con Discapacidad",
                    "relevancia": "Marco internacional de referencia"
                }
            },
            "universidades_internacionales": {
                "gallaudet": {
                    "nombre": "Gallaudet University (EE.UU.)",
                    "especialidad": "Universidad especializada en educación para sordos",
                    "oportunidad": "Colaboración académica internacional"
                },
                "rochester": {
                    "nombre": "Rochester Institute of Technology",
                    "programa": "National Technical Institute for the Deaf",
                    "oportunidad": "Intercambio tecnológico"
                }
            },
            "empresas_tecnologia": {
                "google": {
                    "programa": "Google AI for Social Good",
                    "relevancia": "Recursos tecnológicos y reconocimiento"
                },
                "microsoft": {
                    "programa": "Microsoft AI for Accessibility",
                    "relevancia": "Herramientas y plataforma"
                }
            }
        }
        return international
    
    def create_contact_templates(self):
        """Plantillas de comunicación"""
        templates = {
            "email_inicial_fenasec": {
                "asunto": "Propuesta de Colaboración: Tecnología LSE Ecuador",
                "estructura": [
                    "Saludo respetuoso",
                    "Presentación personal y del proyecto",
                    "Logros actuales (90.26% precisión)",
                    "Beneficios específicos para FENASEC",
                    "Solicitud de reunión",
                    "Adjuntos relevantes",
                    "Despedida cordial"
                ],
                "tono": "Formal pero cercano, mostrando respeto por la comunidad sorda"
            },
            "carta_formal_conadis": {
                "formato": "Membrete oficial universitario",
                "estructura": [
                    "Encabezado institucional",
                    "Referencia a normativas nacionales",
                    "Presentación del proyecto",
                    "Alineación con objetivos CONADIS",
                    "Propuesta de colaboración específica",
                    "Solicitud de reunión técnica",
                    "Firmas institucionales"
                ]
            },
            "propuesta_universitaria": {
                "formato": "Documento académico",
                "secciones": [
                    "Resumen ejecutivo",
                    "Antecedentes y justificación",
                    "Objetivos y metodología",
                    "Recursos necesarios",
                    "Cronograma",
                    "Impacto esperado",
                    "Referencias"
                ]
            }
        }
        return templates
    
    def create_execution_roadmap(self):
        """Hoja de ruta para ejecución"""
        roadmap = {
            "semana_1": {
                "prioridad_1": {
                    "tarea": "Investigar contactos específicos FENASEC",
                    "acciones": [
                        "Visitar sitio web fenasec.org.ec",
                        "Identificar nombres y cargos actuales",
                        "Buscar en redes sociales información reciente",
                        "Contactar Universidad para referencias"
                    ]
                },
                "prioridad_2": {
                    "tarea": "Preparar materiales de presentación",
                    "acciones": [
                        "Video demo del sistema actual",
                        "Documento ejecutivo en español",
                        "Propuesta específica FENASEC",
                        "Presentación PowerPoint"
                    ]
                }
            },
            "semana_2": {
                "prioridad_1": {
                    "tarea": "Contacto inicial FENASEC",
                    "acciones": [
                        "Llamada telefónica o visita presencial",
                        "Envío de propuesta por email",
                        "Seguimiento en redes sociales",
                        "Solicitud formal de reunión"
                    ]
                },
                "prioridad_2": {
                    "tarea": "Contactar CONADIS",
                    "acciones": [
                        "Llamar al 593-2-2459243",
                        "Solicitar cita con Dirección Técnica",
                        "Enviar carta formal",
                        "Preparar propuesta gubernamental"
                    ]
                }
            },
            "semana_3": {
                "prioridad_1": {
                    "tarea": "Seguimiento institucional",
                    "acciones": [
                        "Confirmar reuniones programadas",
                        "Ajustar propuestas según feedback",
                        "Preparar documentación adicional",
                        "Coordinar agendas"
                    ]
                }
            },
            "semana_4": {
                "prioridad_1": {
                    "tarea": "Ejecutar reuniones clave",
                    "acciones": [
                        "Reunión presencial FENASEC",
                        "Presentación en CONADIS",
                        "Seguimiento universidades",
                        "Evaluación de respuestas"
                    ]
                }
            }
        }
        return roadmap
    
    def save_contact_guide(self):
        """Guardar guía completa de contactos"""
        print("📋 Compilando guía de contactos...")
        
        contact_guide = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "purpose": "Guía práctica para ejecutar plan mejora LSE Ecuador",
                "priority": "Documento de alta prioridad para ejecución inmediata"
            },
            "estrategia_fenasec": self.create_fenasec_approach(),
            "contactos_universitarios": self.create_university_contacts(),
            "contactos_gubernamentales": self.create_government_contacts(),
            "contactos_internacionales": self.create_international_contacts(),
            "plantillas_comunicacion": self.create_contact_templates(),
            "hoja_de_ruta": self.create_execution_roadmap(),
            "notas_importantes": [
                "🎯 FENASEC es el contacto MÁS CRÍTICO - sin su apoyo el proyecto no tendrá legitimidad",
                "📞 CONADIS (593-2-2459243) debe ser contactado dentro de las primeras 2 semanas",
                "🎓 Universidad Técnica de Ambato ya conoce el proyecto - usar como referencia",
                "⏰ Timing crucial: contactar antes de que termine el período académico",
                "💡 Sempre presentar beneficios específicos, no solo características técnicas",
                "🤝 Buscar aliados internos en cada organización desde el primer contacto"
            ],
            "checklist_ejecucion": [
                "☐ Investigar nombres actuales FENASEC",
                "☐ Preparar video demo 5 minutos",
                "☐ Redactar propuesta FENASEC específica",
                "☐ Llamar CONADIS 593-2-2459243",
                "☐ Contactar biblioteca UTA biblioteca@uta.edu.ec",
                "☐ Preparar presentación PowerPoint",
                "☐ Programar reunión presencial FENASEC",
                "☐ Enviar carta formal CONADIS",
                "☐ Seguimiento semanal todos los contactos"
            ]
        }
        
        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lse_contact_strategy_{timestamp}.json"
        filepath = os.path.join("analysis", filename)
        
        os.makedirs("analysis", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(contact_guide, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath, contact_guide

def main():
    """Función principal"""
    print("🇪🇨 GUÍA DE CONTACTOS LSE ECUADOR")
    print("=" * 40)
    
    contact_strategy = LSEContactStrategy()
    guide_file, guide = contact_strategy.save_contact_guide()
    
    # Mostrar información crítica
    print("\n🚨 CONTACTOS PRIORITARIOS")
    print("=" * 28)
    print("1️⃣ FENASEC - fenasec.org.ec")
    print("   📍 Quito - Visita presencial OBLIGATORIA")
    print("   🎯 Validación oficial comunidad sorda")
    print("")
    print("2️⃣ CONADIS - 📞 593-2-2459243")
    print("   📍 Quito - Solicitar Dirección Técnica")
    print("   🎯 Respaldo gubernamental")
    print("")
    print("3️⃣ UTA Biblioteca - 📧 biblioteca@uta.edu.ec")
    print("   📍 Ambato - Base académica del proyecto")
    print("   🎯 Recursos universitarios")
    
    print("\n⚡ ACCIONES INMEDIATAS (Esta semana)")
    print("=" * 40)
    roadmap = guide["hoja_de_ruta"]["semana_1"]
    for priority, task in roadmap.items():
        print(f"\n{priority.upper()}: {task['tarea']}")
        for action in task['acciones']:
            print(f"  ✅ {action}")
    
    print("\n📋 CHECKLIST DE EJECUCIÓN")
    print("=" * 30)
    for item in guide["checklist_ejecucion"]:
        print(f"  {item}")
    
    print("\n🎯 NOTAS CRÍTICAS")
    print("=" * 20)
    for note in guide["notas_importantes"][:3]:  # Mostrar solo las 3 más importantes
        print(f"  {note}")
    
    print(f"\n✅ Guía completa guardada en: {guide_file}")
    print("\n🚀 ¡Todo listo para comenzar los contactos!")
    print("\n💡 PRÓXIMO PASO: Llamar a FENASEC o visitarlos en Quito")

if __name__ == "__main__":
    main()
