"""
Servicio de Segmentación Semántica
Usa MediaPipe Selfie Segmentation para segmentar manos pixel por pixel
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class SemanticSegmentationService:
    """Segmentación semántica de manos usando MediaPipe"""
    
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # También usamos detección de manos para landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def segment_hands(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segmenta las manos pixel por pixel del fondo
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Tuple de (imagen segmentada, máscara binaria)
        """
        # ---- Preprocesado para cámaras de baja calidad ----
        # Upscale if very small
        h, w = image.shape[:2]
        if w < 320 or h < 240:
            scale_x = max(1, 640 // w)
            scale_y = max(1, 480 // h)
            image = cv2.resize(image, (w*scale_x, h*scale_y), interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

        # Convert to LAB and apply CLAHE to L channel for contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2,a,b))
        rgb_image = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

        # Procesar segmentación con MediaPipe
        results = self.segmenter.process(rgb_image)

        # Crear máscara de segmentación y limpiar ruido
        mask = results.segmentation_mask
        mask_bin = (mask > 0.45).astype(np.uint8)  # umbral más permisivo

        # Morphological operations to remove small blobs
        kernel = np.ones((5,5), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep largest connected components (likely hands)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            new_mask = np.zeros_like(labels, dtype=np.uint8)
            new_mask[labels == largest_idx] = 255
            # Also keep second largest if reasonably large
            if len(areas) > 1:
                second_idx = np.argsort(areas)[-2] + 1
                if areas[second_idx-1] > 0.3 * areas[largest_idx-1]:
                    new_mask[labels == second_idx] = 255
            binary_mask = new_mask
        else:
            binary_mask = (mask_bin * 255).astype(np.uint8)

        # Crear imagen overlay: mantener manos y desaturar fondo
        overlay = image.copy()
        background = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        # Blend where mask is present
        condition = binary_mask.astype(bool)
        output_image = background.copy()
        output_image[condition] = image[condition]

        return output_image, binary_mask
    
    def extract_hand_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae solo la región de las manos usando la máscara
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Imagen con solo las manos, fondo negro
        """
        _, mask = self.segment_hands(image)
        
        # Aplicar máscara a la imagen original
        hand_region = cv2.bitwise_and(image, image, mask=mask)
        
        return hand_region
    
    def get_hand_landmarks(self, image: np.ndarray) -> Optional[dict]:
        """
        Obtiene landmarks de las manos detectadas
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Dict con landmarks normalizados
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks_data = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer coordenadas de 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Normalizar landmarks (centrar en muñeca)
            landmarks = np.array(landmarks)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            
            # Escalar
            scale = np.linalg.norm(landmarks[12])  # Distancia a dedo medio
            if scale > 0:
                landmarks = landmarks / scale
            
            landmarks_data.append(landmarks.flatten().tolist())
        
        return {
            "num_hands": len(landmarks_data),
            "landmarks": landmarks_data
        }
    
    def process_frame(self, image: np.ndarray) -> dict:
        """
        Procesa un frame completo: segmentación + landmarks
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            Dict con toda la información procesada
        """
        # Segmentación
        segmented, mask = self.segment_hands(image)
        hand_region = self.extract_hand_region(image)
        
        # Landmarks
        landmarks = self.get_hand_landmarks(image)
        
        return {
            "segmented_image": segmented,
            "mask": mask,
            "hand_region": hand_region,
            "landmarks": landmarks,
            "has_hands": landmarks is not None
        }
    
    def __del__(self):
        """Liberar recursos"""
        self.segmenter.close()
        self.hands.close()

# Instancia global del servicio
segmentation_service = SemanticSegmentationService()
