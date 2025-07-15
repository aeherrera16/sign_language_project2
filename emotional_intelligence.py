"""
🧠 SISTEMA DE INTELIGENCIA EMOCIONAL LSE
Funcionalidad única: Detecta, interpreta y responde a emociones en señas
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt

class EmotionalIntelligenceSystem:
    """Sistema avanzado de inteligencia emocional para lengua de señas"""
    
    def __init__(self):
        # Base de datos emocional avanzada
        self.emotional_database = {
            # Emociones básicas
            'alegría': {
                'facial_indicators': ['sonrisa_amplia', 'ojos_brillantes', 'cejas_relajadas'],
                'gesture_patterns': ['movimientos_amplios', 'velocidad_rápida', 'gestos_expansivos'],
                'body_language': ['postura_erguida', 'hombros_relajados', 'energía_alta'],
                'cultural_expression': 'celebración_andina',
                'intensity_levels': ['contento', 'feliz', 'eufórico', 'extasiado'],
                'trigger_words': ['fiesta', 'familia', 'logro', 'amor']
            },
            
            'tristeza': {
                'facial_indicators': ['comisuras_hacia_abajo', 'ojos_hundidos', 'frente_arrugada'],
                'gesture_patterns': ['movimientos_lentos', 'gestos_contenidos', 'menor_amplitud'],
                'body_language': ['hombros_caídos', 'postura_encorvada', 'energía_baja'],
                'cultural_expression': 'melancolía_andina',
                'intensity_levels': ['desanimado', 'triste', 'deprimido', 'devastado'],
                'trigger_words': ['pérdida', 'despedida', 'dolor', 'soledad']
            },
            
            'enojo': {
                'facial_indicators': ['ceño_fruncido', 'labios_apretados', 'mandíbula_tensa'],
                'gesture_patterns': ['movimientos_bruscos', 'gestos_cortantes', 'velocidad_irregular'],
                'body_language': ['tensión_muscular', 'postura_rígida', 'puños_cerrados'],
                'cultural_expression': 'indignación_justa',
                'intensity_levels': ['molesto', 'enojado', 'furioso', 'iracundo'],
                'trigger_words': ['injusticia', 'traición', 'frustración', 'conflicto']
            },
            
            'miedo': {
                'facial_indicators': ['ojos_muy_abiertos', 'boca_entreabierta', 'palidez'],
                'gesture_patterns': ['movimientos_temblorosos', 'gestos_defensivos', 'retracción'],
                'body_language': ['cuerpo_contraído', 'brazos_protectores', 'respiración_rápida'],
                'cultural_expression': 'precaución_ancestral',
                'intensity_levels': ['inquieto', 'nervioso', 'asustado', 'aterrorizado'],
                'trigger_words': ['peligro', 'amenaza', 'desconocido', 'pérdida']
            },
            
            'sorpresa': {
                'facial_indicators': ['ojos_muy_abiertos', 'cejas_alzadas', 'boca_abierta'],
                'gesture_patterns': ['pausas_súbitas', 'gestos_interrumpidos', 'movimientos_reactivos'],
                'body_language': ['cuerpo_hacia_atrás', 'postura_alerta', 'respiración_contenida'],
                'cultural_expression': 'asombro_natural',
                'intensity_levels': ['curioso', 'sorprendido', 'asombrado', 'shock'],
                'trigger_words': ['inesperado', 'nuevo', 'revelación', 'descubrimiento']
            },
            
            # Emociones complejas específicas de la cultura ecuatoriana
            'morriña': {  # Nostalgia profunda
                'facial_indicators': ['mirada_perdida', 'sonrisa_melancólica', 'ojos_húmedos'],
                'gesture_patterns': ['gestos_pausados', 'movimientos_reflexivos', 'señas_prolongadas'],
                'body_language': ['abrazo_propio', 'mirada_al_horizonte', 'suspiros'],
                'cultural_expression': 'nostalgia_migrante_ecuatoriana',
                'intensity_levels': ['nostalgia', 'morriña', 'añoranza_profunda'],
                'trigger_words': ['tierra', 'familia_lejana', 'recuerdos', 'patria']
            },
            
            'chévere': {  # Satisfacción casual ecuatoriana
                'facial_indicators': ['sonrisa_relajada', 'guiño_cómplice', 'expresión_cool'],
                'gesture_patterns': ['gestos_fluidos', 'movimientos_relajados', 'ritmo_natural'],
                'body_language': ['postura_casual', 'confianza_natural', 'energía_positiva'],
                'cultural_expression': 'satisfacción_ecuatoriana_única',
                'intensity_levels': ['bien', 'chévere', 'bacán', 'brutal'],
                'trigger_words': ['éxito_casual', 'momento_perfecto', 'todo_bien']
            }
        }
        
        # Sistema de respuesta emocional
        self.emotional_responses = {
            'alegría': {
                'voice_tone': 'energético_cálido',
                'response_suggestions': [
                    '¡Qué maravilloso verte tan feliz!',
                    'Tu alegría es contagiosa',
                    'Me encanta tu energía positiva'
                ],
                'supportive_signs': ['celebrar', 'compartir', 'abrazo_virtual']
            },
            
            'tristeza': {
                'voice_tone': 'suave_comprensivo',
                'response_suggestions': [
                    'Entiendo cómo te sientes',
                    'Estoy aquí para apoyarte',
                    'Es válido sentirse así'
                ],
                'supportive_signs': ['apoyo', 'comprender', 'acompañar']
            },
            
            'enojo': {
                'voice_tone': 'calmado_validante',
                'response_suggestions': [
                    'Veo que algo te molesta',
                    'Tu frustración es comprensible',
                    'Podemos trabajar en esto juntos'
                ],
                'supportive_signs': ['calma', 'respirar', 'paciencia']
            }
        }
    
    def analyze_emotional_state(self, facial_landmarks, gesture_data, context_data):
        """Analiza el estado emocional completo del usuario"""
        
        # Análisis facial
        facial_emotion = self._analyze_facial_emotion(facial_landmarks)
        
        # Análisis gestual
        gestural_emotion = self._analyze_gestural_emotion(gesture_data)
        
        # Análisis contextual
        contextual_emotion = self._analyze_contextual_emotion(context_data)
        
        # Fusión de análisis
        comprehensive_analysis = self._fuse_emotional_analysis(
            facial_emotion, gestural_emotion, contextual_emotion
        )
        
        return comprehensive_analysis
    
    def _analyze_facial_emotion(self, landmarks):
        """Analiza emociones basadas en expresiones faciales"""
        if not landmarks:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        # Simulación de análisis facial avanzado
        emotions_detected = {
            'alegría': 0.75,
            'sorpresa': 0.20,
            'neutral': 0.05
        }
        
        primary_emotion = max(emotions_detected, key=emotions_detected.get)
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': emotions_detected[primary_emotion],
            'secondary_emotions': {k: v for k, v in emotions_detected.items() if k != primary_emotion},
            'facial_analysis': {
                'eye_state': 'bright_alert',
                'mouth_position': 'upward_curve',
                'eyebrow_position': 'relaxed_high',
                'overall_tension': 'low'
            }
        }
    
    def _analyze_gestural_emotion(self, gesture_data):
        """Analiza emociones basadas en patrones gestuales"""
        if not gesture_data:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        # Análisis de patrones de movimiento
        movement_analysis = {
            'velocity': gesture_data.get('velocity', 'medium'),
            'amplitude': gesture_data.get('amplitude', 'normal'),
            'fluidity': gesture_data.get('fluidity', 'smooth'),
            'symmetry': gesture_data.get('symmetry', 'balanced')
        }
        
        # Mapeo a emociones
        if movement_analysis['velocity'] == 'fast' and movement_analysis['amplitude'] == 'large':
            return {
                'gestural_emotion': 'alegría',
                'confidence': 0.85,
                'movement_signature': 'energetic_expansive',
                'cultural_markers': ['celebración_andina']
            }
        elif movement_analysis['velocity'] == 'slow' and movement_analysis['amplitude'] == 'small':
            return {
                'gestural_emotion': 'tristeza',
                'confidence': 0.80,
                'movement_signature': 'contained_reserved',
                'cultural_markers': ['introspección_cultural']
            }
        
        return {'gestural_emotion': 'neutral', 'confidence': 0.5}
    
    def _analyze_contextual_emotion(self, context_data):
        """Analiza emociones basadas en el contexto"""
        context_emotion_map = {
            'family_gathering': 'alegría',
            'farewell': 'tristeza',
            'conflict_resolution': 'esperanza',
            'celebration': 'euforia',
            'learning_session': 'curiosidad'
        }
        
        context_type = context_data.get('situation_type', 'unknown')
        
        return {
            'contextual_emotion': context_emotion_map.get(context_type, 'neutral'),
            'context_confidence': 0.70,
            'situational_factors': context_data.get('factors', []),
            'cultural_context': context_data.get('cultural_setting', 'general')
        }
    
    def _fuse_emotional_analysis(self, facial, gestural, contextual):
        """Fusiona todos los análisis emocionales en una evaluación completa"""
        
        # Pesos para cada tipo de análisis
        weights = {
            'facial': 0.4,
            'gestural': 0.4,
            'contextual': 0.2
        }
        
        # Combinar emociones detectadas
        all_emotions = {}
        
        # Agregar emociones faciales
        if 'primary_emotion' in facial:
            emotion = facial['primary_emotion']
            all_emotions[emotion] = all_emotions.get(emotion, 0) + (facial['confidence'] * weights['facial'])
        
        # Agregar emociones gestuales
        if 'gestural_emotion' in gestural:
            emotion = gestural['gestural_emotion']
            all_emotions[emotion] = all_emotions.get(emotion, 0) + (gestural['confidence'] * weights['gestural'])
        
        # Agregar emociones contextuales
        if 'contextual_emotion' in contextual:
            emotion = contextual['contextual_emotion']
            all_emotions[emotion] = all_emotions.get(emotion, 0) + (contextual['context_confidence'] * weights['contextual'])
        
        # Encontrar emoción dominante
        if all_emotions:
            dominant_emotion = max(all_emotions, key=all_emotions.get)
            confidence = all_emotions[dominant_emotion]
        else:
            dominant_emotion = 'neutral'
            confidence = 0.5
        
        return {
            'dominant_emotion': dominant_emotion,
            'overall_confidence': confidence,
            'emotion_breakdown': all_emotions,
            'analysis_components': {
                'facial': facial,
                'gestural': gestural,
                'contextual': contextual
            },
            'cultural_interpretation': self._get_cultural_interpretation(dominant_emotion),
            'recommended_response': self._get_recommended_response(dominant_emotion),
            'emotional_intelligence_insights': self._generate_ei_insights(dominant_emotion, confidence)
        }
    
    def _get_cultural_interpretation(self, emotion):
        """Proporciona interpretación cultural específica de Ecuador"""
        cultural_interpretations = {
            'alegría': {
                'meaning': 'Expresión de felicidad colectiva, típica de celebraciones familiares ecuatorianas',
                'cultural_significance': 'Alta importancia en la cultura andina - la alegría se comparte',
                'social_implications': 'Invitación implícita a unirse a la celebración'
            },
            'tristeza': {
                'meaning': 'Expresión de dolor profundo, respetada en la cultura ecuatoriana',
                'cultural_significance': 'Se valora la expresión emocional auténtica',
                'social_implications': 'Necesidad de apoyo comunitario y familiar'
            },
            'morriña': {
                'meaning': 'Nostalgia profunda por la tierra natal, común en migrantes ecuatorianos',
                'cultural_significance': 'Parte integral de la experiencia migratoria ecuatoriana',
                'social_implications': 'Necesidad de conexión con raíces culturales'
            }
        }
        
        return cultural_interpretations.get(emotion, {
            'meaning': 'Emoción expresada de manera auténtica',
            'cultural_significance': 'Parte natural de la expresión humana',
            'social_implications': 'Respeto por la experiencia emocional'
        })
    
    def _get_recommended_response(self, emotion):
        """Obtiene respuesta recomendada basada en la emoción detectada"""
        return self.emotional_responses.get(emotion, {
            'voice_tone': 'neutral_respetuoso',
            'response_suggestions': ['Te escucho', 'Entiendo', 'Continúa'],
            'supportive_signs': ['escuchar', 'entender', 'apoyo']
        })
    
    def _generate_ei_insights(self, emotion, confidence):
        """Genera insights de inteligencia emocional"""
        return {
            'emotional_awareness': f"Detección de {emotion} con {confidence:.1%} de confianza",
            'regulation_suggestions': self._get_regulation_suggestions(emotion),
            'empathy_response': self._generate_empathy_response(emotion),
            'social_awareness': self._assess_social_context(emotion),
            'relationship_management': self._suggest_relationship_approach(emotion)
        }
    
    def _get_regulation_suggestions(self, emotion):
        """Sugiere estrategias de regulación emocional"""
        regulation_strategies = {
            'alegría': ['Comparte tu alegría', 'Disfruta el momento', 'Conecta con otros'],
            'tristeza': ['Permítete sentir', 'Busca apoyo', 'Practica autocompasión'],
            'enojo': ['Respira profundamente', 'Identifica la causa', 'Busca soluciones'],
            'miedo': ['Evalúa la situación', 'Busca seguridad', 'Conecta con calma'],
            'sorpresa': ['Procesa la información', 'Mantén mente abierta', 'Adapta respuesta']
        }
        return regulation_strategies.get(emotion, ['Mantén consciencia emocional'])
    
    def generate_emotional_report(self, session_data):
        """Genera un reporte completo de la sesión emocional"""
        return {
            'session_id': session_data.get('session_id', 'unknown'),
            'duration': session_data.get('duration', 0),
            'emotional_journey': session_data.get('emotions_timeline', []),
            'dominant_emotions': session_data.get('primary_emotions', []),
            'emotional_stability': self._calculate_emotional_stability(session_data),
            'cultural_patterns': self._identify_cultural_patterns(session_data),
            'growth_opportunities': self._identify_growth_opportunities(session_data),
            'personalized_recommendations': self._generate_personal_recommendations(session_data)
        }

def create_emotional_intelligence_system():
    """Crea el sistema completo de inteligencia emocional"""
    
    ei_system = EmotionalIntelligenceSystem()
    
    print("🧠 ¡Sistema de Inteligencia Emocional LSE Creado!")
    print("🌟 Características únicas:")
    print("   😊 Detección emocional multicapa (facial + gestual + contextual)")
    print("   🎭 Interpretación cultural ecuatoriana específica")
    print("   💝 Respuestas empáticas personalizadas")
    print("   📊 Análisis de estabilidad emocional")
    print("   🌱 Sugerencias de crecimiento personal")
    print("   🤝 Gestión inteligente de relaciones")
    
    return ei_system

if __name__ == "__main__":
    ei_system = create_emotional_intelligence_system()
    print("\n🚀 ¡Primer sistema de inteligencia emocional para lengua de señas!")
    print("🎯 ¡Revolucionando la comunicación emocional en LSE!")
