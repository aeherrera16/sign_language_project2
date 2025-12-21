"""
Servicio Avanzado de Segmentación de Manos
Usa MediaPipe Hands para enfocarse exclusivamente en las manos
con segmentación semántica pixel a pixel basada en color de piel
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class HandRegion:
    """Información de una región de mano detectada"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray
    mask: np.ndarray
    cropped_image: np.ndarray
    confidence: float


class HandSegmentationService:
    """
    Servicio avanzado de segmentación enfocado SOLO en manos
    
    Features:
    - MediaPipe Hands para detección precisa
    - Segmentación semántica de piel (HSV + LAB)
    - Recorte inteligente con padding
    - Análisis de calidad con IA
    """
    
    def __init__(self):
        # MediaPipe Hands - SOLO detecta manos, no cuerpo completo
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Rangos de color para segmentación de piel - MÁS INCLUSIVOS
        # HSV ranges for skin detection (ampliados para todos los tonos de piel)
        self.hsv_lower = np.array([0, 10, 50], dtype=np.uint8)  # Más permisivo
        self.hsv_upper = np.array([25, 255, 255], dtype=np.uint8)  # Extendido a 25
        self.hsv_lower2 = np.array([165, 10, 50], dtype=np.uint8)  # Rojo wrap-around
        self.hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # LAB ranges for better lighting invariance - AMPLIADOS
        self.lab_l_range = (20, 240)  # Luminosidad amplia
        self.lab_a_range = (120, 180)  # Más amplio para diferentes tonos
        self.lab_b_range = (115, 185)  # Más amplio
        
    def detect_hands(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detectar manos usando MediaPipe Hands
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Dict con información de manos detectadas o None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        h, w = image.shape[:2]
        hands_info = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extraer landmarks en coordenadas de pixel
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            landmarks = np.array(landmarks)
            
            # Calcular bounding box con padding
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            
            padding = 40  # pixels de margen
            x_min = max(0, int(np.min(x_coords)) - padding)
            y_min = max(0, int(np.min(y_coords)) - padding)
            x_max = min(w, int(np.max(x_coords)) + padding)
            y_max = min(h, int(np.max(y_coords)) + padding)
            
            # Info del tipo de mano
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
            
            hands_info.append({
                'landmarks': landmarks,
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                'handedness': handedness,
                'confidence': results.multi_handedness[idx].classification[0].score if results.multi_handedness else 0.5
            })
        
        return {
            'num_hands': len(hands_info),
            'hands': hands_info
        }
    
    def segment_skin_pixels(self, image: np.ndarray, mask_region: np.ndarray = None) -> np.ndarray:
        """
        Segmentar piel pixel a pixel usando HSV + LAB
        
        Args:
            image: Imagen BGR
            mask_region: Máscara opcional para limitar área de búsqueda
            
        Returns:
            Máscara binaria de piel
        """
        # Convertir a HSV y LAB
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Máscara HSV (dos rangos para cubrir rojo que cruza 0/180)
        mask_hsv1 = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        
        # Máscara LAB para mejor invarianza a iluminación
        l, a, b = cv2.split(lab)
        mask_lab = np.zeros_like(l)
        mask_lab[(l >= self.lab_l_range[0]) & (l <= self.lab_l_range[1]) &
                 (a >= self.lab_a_range[0]) & (a <= self.lab_a_range[1]) &
                 (b >= self.lab_b_range[0]) & (b <= self.lab_b_range[1])] = 255
        
        # Combinar máscaras
        combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
        
        # Aplicar región de interés si se proporciona
        if mask_region is not None:
            combined_mask = cv2.bitwise_and(combined_mask, mask_region)
        
        # Limpieza morfológica
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Suavizar bordes
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        return combined_mask
    
    def create_hand_mask_from_landmarks(self, image_shape: Tuple, landmarks: np.ndarray) -> np.ndarray:
        """
        Crear máscara convexa a partir de landmarks de la mano
        
        Args:
            image_shape: Forma de la imagen (h, w)
            landmarks: Array de landmarks [21, 3]
            
        Returns:
            Máscara binaria
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Crear convex hull de los landmarks
            points = landmarks[:, :2].astype(np.int32)
            hull = cv2.convexHull(points)
            
            # Reshape hull para manejar correctamente - hull tiene shape (N, 1, 2)
            hull_points = hull.reshape(-1, 2)
            
            # Calcular centro
            center = np.mean(hull_points, axis=0)
            
            # Expandir el hull un poco para cubrir toda la mano
            expanded_hull = []
            for point in hull_points:
                direction = point - center
                expansion = direction * 0.20  # 20% de expansión
                new_point = point + expansion
                # Clamp to image bounds
                new_x = int(np.clip(new_point[0], 0, w - 1))
                new_y = int(np.clip(new_point[1], 0, h - 1))
                expanded_hull.append([new_x, new_y])
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            cv2.fillConvexPoly(mask, expanded_hull, 255)
            
        except Exception as e:
            # Fallback: usar los landmarks directamente sin expansión
            try:
                points = landmarks[:, :2].astype(np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
            except:
                pass  # Si todo falla, devolver máscara vacía
        
        return mask
    
    def segment_hands_only(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        FUNCIÓN PRINCIPAL: Segmentar SOLO las manos de la imagen
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Tuple de (imagen_solo_manos, máscara_combinada, métricas)
        """
        h, w = image.shape[:2]
        
        # 1. Detectar manos con MediaPipe
        hands_info = self.detect_hands(image)
        
        if not hands_info or hands_info['num_hands'] == 0:
            return image, np.zeros((h, w), dtype=np.uint8), {
                'hands_detected': 0,
                'hands_percentage': 0,
                'quality_score': 0,
                'hands_info': [],
                'message': 'No se detectaron manos'
            }
        
        # 2. Crear máscara combinada de todas las manos
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for hand in hands_info['hands']:
            # Máscara del convex hull
            hull_mask = self.create_hand_mask_from_landmarks(image.shape, hand['landmarks'])
            
            # Segmentación de piel dentro del área del hull
            skin_mask = self.segment_skin_pixels(image, hull_mask)
            
            # Combinar
            combined_mask = cv2.bitwise_or(combined_mask, skin_mask)
        
        # 3. Refinar máscara final
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Crear imagen con solo las manos (fondo negro)
        hands_only = np.zeros_like(image)
        hands_only[combined_mask > 0] = image[combined_mask > 0]
        
        # 5. Calcular métricas
        hands_pixels = np.sum(combined_mask > 0)
        total_pixels = h * w
        hands_percentage = (hands_pixels / total_pixels) * 100
        
        # Calidad basada en detección y porcentaje
        base_score = min(100, hands_percentage * 5)  # Normalizar
        confidence_boost = np.mean([h['confidence'] for h in hands_info['hands']]) * 20
        quality_score = min(100, base_score + confidence_boost)
        
        metrics = {
            'hands_detected': hands_info['num_hands'],
            'hands_percentage': round(hands_percentage, 2),
            'quality_score': round(quality_score, 2),
            'hands_info': [{'handedness': h['handedness'], 'confidence': h['confidence']} 
                          for h in hands_info['hands']],
            'message': 'OK' if hands_info['num_hands'] > 0 else 'Sin manos'
        }
        
        return hands_only, combined_mask, metrics
    
    def get_cropped_hands(self, image: np.ndarray, padding: int = 50) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Obtener imagen recortada centrada en las manos
        
        Args:
            image: Imagen BGR
            padding: Pixels de margen alrededor de las manos
            
        Returns:
            Tuple de (imagen_recortada, métricas)
        """
        hands_info = self.detect_hands(image)
        
        if not hands_info or hands_info['num_hands'] == 0:
            return None, {'error': 'No se detectaron manos'}
        
        h, w = image.shape[:2]
        
        # Calcular bounding box que englobe todas las manos
        all_x = []
        all_y = []
        for hand in hands_info['hands']:
            landmarks = hand['landmarks']
            all_x.extend(landmarks[:, 0])
            all_y.extend(landmarks[:, 1])
        
        x_min = max(0, int(min(all_x)) - padding)
        y_min = max(0, int(min(all_y)) - padding)
        x_max = min(w, int(max(all_x)) + padding)
        y_max = min(h, int(max(all_y)) + padding)
        
        # Recortar
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Segmentar solo las manos en la región recortada
        hands_only, mask, metrics = self.segment_hands_only(cropped)
        
        return hands_only, metrics
    
    def draw_hand_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Dibujar landmarks de manos sobre la imagen
        
        Args:
            image: Imagen BGR
            
        Returns:
            Imagen con landmarks dibujados
        """
        output = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        return output
    
    def compute_quality_score(self, image: np.ndarray) -> Dict:
        """
        Calcular puntuación de calidad para captura
        
        Args:
            image: Imagen BGR
            
        Returns:
            Dict con métricas de calidad
        """
        _, _, metrics = self.segment_hands_only(image)
        
        # Calcular nitidez
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        
        # Calcular iluminación
        brightness = np.mean(gray)
        lighting_score = 100 - abs(brightness - 127) * 0.8  # Óptimo en 127
        
        # Score final
        final_score = (
            metrics['quality_score'] * 0.5 +
            sharpness_score * 0.3 +
            lighting_score * 0.2
        )
        
        is_good = (
            metrics['hands_detected'] >= 1 and
            final_score >= 60 and
            sharpness_score >= 30
        )
        
        # Generar recomendaciones
        recommendations = []
        if metrics['hands_detected'] == 0:
            recommendations.append('❌ No se detectan manos - asegúrate de que estén visibles')
        elif metrics['hands_detected'] == 1:
            recommendations.append('⚠️ Solo una mano detectada')
        else:
            recommendations.append('✅ Ambas manos detectadas')
        
        if metrics['hands_percentage'] < 5:
            recommendations.append('❌ Acerca más las manos a la cámara')
        elif metrics['hands_percentage'] < 10:
            recommendations.append('⚠️ Acerca un poco más las manos')
        else:
            recommendations.append('✅ Tamaño de manos adecuado')
        
        if sharpness_score < 30:
            recommendations.append('❌ Imagen borrosa - muévete más lento')
        elif sharpness_score < 50:
            recommendations.append('⚠️ Mejora la nitidez')
        else:
            recommendations.append('✅ Imagen nítida')
        
        if lighting_score < 50:
            recommendations.append('⚠️ Mejora la iluminación')
        
        return {
            'final_score': round(final_score, 2),
            'hands_score': metrics['quality_score'],
            'sharpness_score': round(sharpness_score, 2),
            'lighting_score': round(lighting_score, 2),
            'hands_detected': metrics['hands_detected'],
            'hands_percentage': metrics['hands_percentage'],
            'is_good': is_good,
            'recommendations': recommendations[:4]
        }
    
    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'hands'):
            self.hands.close()


# Instancia global del servicio
hand_segmentation_service = HandSegmentationService()
