"""
Servicio de integración con LLM usando Ollama (GRATIS).
Compatible con DeepSeek R1, Llama, Mistral, etc.

Este servicio permite:
- Analizar el contexto de una seña reconocida
- Mejorar el reconocimiento con información contextual
- Generar feedback educativo para el usuario

🆓 100% GRATIS - No requiere API key ni pagar tokens
🔌 OFFLINE - Funciona sin internet
⚡ RÁPIDO - Respuestas en segundos
"""

import httpx
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json

load_dotenv()


class LLMContextService:
    """
    Servicio para integrar LLM local con Ollama.
    NO REQUIERE API KEY - TODO GRATIS.
    """
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "not_needed")  # No se usa con Ollama
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("LLM_MODEL", "deepseek-r1")
        
    async def _call_llm(self, messages: List[dict]) -> str:
        """
        Realiza una llamada al LLM usando Ollama (GRATIS).
        Compatible con DeepSeek R1, Llama, Mistral, etc.
        """
        try:
            # Ollama usa endpoint /api/chat (NO /chat/completions)
            data = {
                "model": self.model,
                "messages": messages,
                "stream": False  # Respuesta completa, no streaming
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",  # Endpoint de Ollama
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                # Ollama retorna en formato diferente a OpenAI
                return result["message"]["content"]
                
        except httpx.ConnectError:
            return "❌ Ollama no está corriendo. Ejecuta: ollama serve"
        except Exception as e:
            print(f"Error en llamada LLM: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_gesture_context(
        self, 
        detected_gesture: str, 
        confidence: float,
        previous_gestures: List[str] = None
    ) -> Dict:
        """
        Analiza el contexto de una seña detectada usando el LLM.
        
        Args:
            detected_gesture: Seña detectada por el modelo
            confidence: Nivel de confianza de la detección (0-1)
            previous_gestures: Lista de señas previas en la conversación
            
        Returns:
            Dict con análisis contextual y sugerencias
        """
        # Construir contexto previo
        previous_context = ""
        if previous_gestures:
            previous_context = f"Señas previas: {', '.join(previous_gestures[-5:])}\n"
        
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en Lengua de Señas Ecuatoriana (LSE). Responde en formato JSON válido."
            },
            {
                "role": "user",
                "content": f"""Analiza esta seña:

{previous_context}
Seña detectada: "{detected_gesture}"
Confianza: {confidence:.2%}

Responde en JSON:
{{
  "context": "análisis breve del contexto",
  "likely_meaning": "significado más probable",
  "similar_signs": ["seña1", "seña2"],
  "confidence_adjusted": 0.95
}}"""
            }
        ]
        
        response = await self._call_llm(messages)
        
        # Intentar parsear JSON
        try:
            result = json.loads(response)
            return result
        except:
            # Si no es JSON válido, retornar formato default
            return {
                "context": response,
                "likely_meaning": detected_gesture,
                "similar_signs": [],
                "confidence_adjusted": confidence
            }
    
    async def improve_recognition(
        self,
        top_predictions: List[Dict[str, float]],
        image_description: str = ""
    ) -> Dict:
        """
        Usa el LLM para mejorar el reconocimiento analizando múltiples predicciones.
        
        Args:
            top_predictions: Lista de predicciones [{"gesture": "hola", "confidence": 0.85}, ...]
            image_description: Descripción textual de lo que se ve (opcional)
            
        Returns:
            Predicción mejorada con razonamiento
        """
        if not top_predictions:
            return {"gesture": "desconocido", "confidence": 0.0, "reasoning": "Sin predicciones"}
        
        predictions_text = "\n".join([
            f"- {p['gesture']}: {p['confidence']:.2%}"
            for p in top_predictions[:5]
        ])
        
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en Lengua de Señas Ecuatoriana (LSE)."
            },
            {
                "role": "user",
                "content": f"""Analiza estas predicciones de reconocimiento de señas:

{image_description}

Top predicciones del modelo:
{predictions_text}

¿Cuál es la más probable considerando el contexto?
Responde con:
1. La seña seleccionada
2. Breve razonamiento (1-2 líneas)"""
            }
        ]
        
        response = await self._call_llm(messages)
        
        return {
            "gesture": top_predictions[0]["gesture"],
            "confidence": top_predictions[0]["confidence"],
            "reasoning": response
        }
    
    async def generate_learning_feedback(
        self,
        gesture_name: str,
        user_performance: float,
        common_mistakes: List[str] = None
    ) -> str:
        """
        Genera feedback educativo para ayudar al usuario a mejorar.
        
        Args:
            gesture_name: Nombre de la seña que está practicando
            user_performance: Nivel de precisión del usuario (0-1)
            common_mistakes: Lista de errores comunes observados
            
        Returns:
            Feedback educativo personalizado
        """
        mistakes_text = ""
        if common_mistakes:
            mistakes_text = f"\nErrores observados: {', '.join(common_mistakes)}"
        
        messages = [
            {
                "role": "system",
                "content": "Eres un profesor experto en Lengua de Señas Ecuatoriana (LSE). Eres motivador y constructivo."
            },
            {
                "role": "user",
                "content": f"""El usuario está practicando la seña: "{gesture_name}"
Precisión actual: {user_performance:.2%}
{mistakes_text}

Genera feedback educativo (2-3 líneas):
- Qué está haciendo bien
- Qué puede mejorar
- Un consejo práctico"""
            }
        ]
        
        response = await self._call_llm(messages)
        return response
    
    async def translate_to_text(
        self,
        gesture_sequence: List[str]
    ) -> str:
        """
        Convierte una secuencia de señas en texto natural en español.
        
        Args:
            gesture_sequence: Lista de señas detectadas en orden
            
        Returns:
            Texto en español natural
        """
        if not gesture_sequence:
            return ""
        
        gestures_text = " + ".join(gesture_sequence)
        
        messages = [
            {
                "role": "system",
                "content": "Eres un intérprete experto de Lengua de Señas Ecuatoriana (LSE) a español."
            },
            {
                "role": "user",
                "content": f"""Traduce esta secuencia de señas LSE a español natural:

{gestures_text}

Responde solo con la traducción en español, sin explicaciones."""
            }
        ]
        
        response = await self._call_llm(messages)
        return response.strip()
    
    async def suggest_next_signs(
        self,
        current_sequence: List[str]
    ) -> List[str]:
        """
        Sugiere las siguientes señas más probables basándose en el contexto.
        
        Args:
            current_sequence: Secuencia actual de señas
            
        Returns:
            Lista de señas sugeridas
        """
        if not current_sequence:
            return ["hola", "buenos días", "gracias"]
        
        sequence_text = " → ".join(current_sequence)
        
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en Lengua de Señas Ecuatoriana (LSE). Responde en formato JSON."
            },
            {
                "role": "user",
                "content": f"""Basándote en esta secuencia de señas LSE:

{sequence_text}

¿Cuáles son las 3 señas más probables que seguirían?

Responde en JSON:
{{
  "suggestions": ["seña1", "seña2", "seña3"]
}}"""
            }
        ]
        
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return result.get("suggestions", [])
        except:
            return []


# Instancia global del servicio
llm_service = LLMContextService()
