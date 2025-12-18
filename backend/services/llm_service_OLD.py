"""
Servicio de IA Generativa para contexto semántico
Usa DeepSeek API (compatible con OpenAI) para mejorar el reconocimiento
"""

import httpx
import os
from typing import List, Dict, Optional
import json

class LLMContextService:
    """Servicio para conectar con DeepSeek u otro LLM compatible con OpenAI"""
    
    def __init__(self):
        # Puede ser DeepSeek, OpenAI, o cualquier API compatible
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        
    async def analyze_gesture_context(
        self, 
        detected_gesture: str, 
        confidence: float,
        previous_gestures: List[str] = None
    ) -> Dict:
        """
        Analiza el contexto de una seña detectada usando el LLM
        
        Args:
            detected_gesture: Seña detectada por el modelo
            confidence: Nivel de confianza de la detección
            previous_gestures: Lista de señas previas en la conversación
            
        Returns:
            Dict con análisis contextual y sugerencias
        """
        if not self.api_key:
            return {
                "context": "No LLM configured",
                "suggestions": [],
                "confidence_adjusted": confidence
            }
        
        # Construir prompt
        previous_context = ""
        if previous_gestures:
            previous_context = f"Señas previas: {', '.join(previous_gestures[-5:])}\n"
        
        prompt = f"""Eres un experto en Lengua de Señas Ecuatoriana (LSE).

{previous_context}
Seña actual detectada: "{detected_gesture}" (confianza: {confidence:.2f})

Analiza:
1. ¿La seña tiene sentido en este contexto?
2. ¿Hay señas similares que podrían confundirse?
3. ¿Qué significado probable tiene en esta conversación?

Responde en formato JSON con:
- context: breve análisis contextual
- likely_meaning: significado más probable
- similar_signs: lista de señas similares
- confidence_adjusted: confianza ajustada (0-1)
"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Eres un experto en Lengua de Señas Ecuatoriana."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 300
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Intentar parsear JSON de la respuesta
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except:
                        return {
                            "context": content,
                            "suggestions": [],
                            "confidence_adjusted": confidence
                        }
                else:
                    return {
                        "context": "Error en API",
                        "suggestions": [],
                        "confidence_adjusted": confidence
                    }
                    
        except Exception as e:
            print(f"Error en LLM service: {e}")
            return {
                "context": f"Error: {str(e)}",
                "suggestions": [],
                "confidence_adjusted": confidence
            }
    
    async def improve_recognition(
        self,
        image_description: str,
        top_predictions: List[Dict[str, float]]
    ) -> Dict:
        """
        Usa el LLM para mejorar el reconocimiento analizando múltiples predicciones
        
        Args:
            image_description: Descripción textual de lo que se ve
            top_predictions: Lista de predicciones top (seña: confianza)
            
        Returns:
            Predicción mejorada con razonamiento
        """
        if not self.api_key:
            return top_predictions[0] if top_predictions else {}
        
        predictions_text = "\n".join([
            f"- {p['gesture']}: {p['confidence']:.2%}"
            for p in top_predictions[:5]
        ])
        
        prompt = f"""Analiza estas predicciones de Lengua de Señas Ecuatoriana:

Observación: {image_description}

Top predicciones:
{predictions_text}

¿Cuál es la más probable considerando el contexto visual y semántico?
Responde con la seña seleccionada y un breve razonamiento."""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 150
                    },
                    timeout=8.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    reasoning = result["choices"][0]["message"]["content"]
                    
                    return {
                        "improved_prediction": top_predictions[0],
                        "reasoning": reasoning
                    }
                    
        except Exception as e:
            print(f"Error en improve_recognition: {e}")
            
        return {
            "improved_prediction": top_predictions[0] if top_predictions else {},
            "reasoning": "Error en análisis LLM"
        }
    
    async def generate_learning_feedback(
        self,
        user_gesture: str,
        model_prediction: str,
        was_correct: bool
    ) -> str:
        """
        Genera feedback educativo para ayudar al usuario
        
        Args:
            user_gesture: La seña que el usuario intentaba hacer
            model_prediction: Lo que el modelo detectó
            was_correct: Si fue correcto o no
            
        Returns:
            Texto con feedback constructivo
        """
        if not self.api_key:
            if was_correct:
                return f"✓ ¡Correcto! Seña '{user_gesture}' reconocida."
            else:
                return f"✗ Se detectó '{model_prediction}' en lugar de '{user_gesture}'. Intenta nuevamente."
        
        status = "correcta" if was_correct else "incorrecta"
        prompt = f"""El usuario intentó hacer la seña "{user_gesture}" en Lengua de Señas Ecuatoriana.
El sistema detectó: "{model_prediction}"
Resultado: {status}

Genera un feedback constructivo y breve (máx 2 líneas) para ayudar al usuario."""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 100
                    },
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            print(f"Error generando feedback: {e}")
        
        if was_correct:
            return f"✓ ¡Bien hecho! Seña '{user_gesture}' reconocida correctamente."
        else:
            return f"Intenta hacer la seña '{user_gesture}' más lentamente y con más claridad."

# Instancia global
llm_service = LLMContextService()
