import os
import sys
import requests
from bs4 import BeautifulSoup
import yt_dlp
import cv2
import numpy as np
import time

# Añadir backend al path para importar servicios
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.hand_segmentation_service import HandSegmentationService

segmentation_service = HandSegmentationService()

BASE_URL = "http://www.plataformaconadis.gob.ec/~platafor/diccionario/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'data', 'gestures')

def get_video_url(article_url):
    """Extrae la URL de YouTube de un artículo"""
    try:
        response = requests.get(article_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        iframe = soup.find('iframe')
        if iframe and 'youtube' in iframe.get('src', ''):
            return iframe['src']
    except Exception as e:
        print(f"Error extrayendo video de {article_url}: {e}")
    return None

def download_video(youtube_url, output_path):
    """Descarga video usando yt-dlp"""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return True
    except Exception as e:
        print(f"Error descargando {youtube_url}: {e}")
        return False

def process_video_to_landmarks(video_path, gesture_name):
    """Convierte el video a secuencia de landmarks"""
    cap = cv2.VideoCapture(video_path)
    sequence_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detectar manos
        hands_info = segmentation_service.detect_hands(frame)
        
        if hands_info and hands_info['num_hands'] > 0:
            frame_landmarks = []
            for hand in hands_info['hands']:
                frame_landmarks.append(hand['landmarks']) # Already normalized
            
            # Rellenar a 2 manos
            while len(frame_landmarks) < 2:
                frame_landmarks.append(np.zeros((21, 3)))
            
            sequence_landmarks.append(np.array(frame_landmarks[:2]))
        else:
            # Si no detecta, agregar frame vacío para mantener temporalidad (o saltar)
            # Para entrenamiento es mejor saltar frames vacíos o interpolar, 
            # pero aquí agregaremos ceros para simplificar
             sequence_landmarks.append(np.zeros((2, 21, 3)))

    cap.release()
    
    # Guardar si tenemos datos útiles
    if len(sequence_landmarks) > 10:
        clean_name = "".join(x for x in gesture_name if x.isalnum())
        save_dir = os.path.join(OUTPUT_DIR, clean_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar secuencia completa
        timestamp = int(time.time())
        np.save(f"{save_dir}/conadis_{timestamp}.npy", np.array(sequence_landmarks))
        
        # Guardar una imagen de referencia (thumbnail)
        # Re-leer video para sacar frame central
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_dir}/reference.jpg", frame)
        cap.release()
        
        print(f"✅ Procesado: {gesture_name} ({len(sequence_landmarks)} frames)")
        return True
    else:
        print(f"⚠️ Video muy corto o sin manos detectadas: {gesture_name}")
        return False

def scrape_category(category_slug, target_letters=None):
    """Scrapea una categoría buscando solo letras específicas"""
    url = f"{BASE_URL}?article_category={category_slug}"
    print(f"🔍 Buscando en: {url} (Objetivo: {target_letters})")
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = soup.find_all('article')
        if not articles:
            # Fallback
            headers = soup.find_all('h2')
            links = []
            for h in headers:
                a = h.find('a')
                if a: links.append(a)
        else:
            links = [a.find('h2').find('a') for a in articles if a.find('h2')]
        
        print(f"Encontrados {len(links)} artículos totales.")
        
        processed_count = 0
        for link in links:
            title = link.text.strip()
            
            # FILTRO: Solo queremos letras individuales "A", "B", o "Letra A"
            clean_title = title.upper().replace("LETRA", "").strip()
            
            # Condición: Longitud 1 (ej "A") o está en nuestra lista objetivo
            is_valid_letter = (len(clean_title) == 1 and clean_title.isalpha())
            
            if target_letters:
                is_valid_letter = clean_title in target_letters
            
            if not is_valid_letter:
                print(f"  ⏭️ Saltando: {title}")
                continue
            
            print(f"\n🎬 ¡ENCONTRADO ABECEDARIO!: {title}")
            href = link['href']
            
            yt_url = get_video_url(href)
            if yt_url:
                # Guardar como "Letra_A", "Letra_B" para evitar conflictos
                final_name = f"Letra_{clean_title}"
                temp_video = f"temp_{clean_title}.mp4"
                
                if download_video(yt_url, temp_video):
                    if process_video_to_landmarks(temp_video, final_name):
                        processed_count += 1
                        
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
            else:
                print("❌ No se encontró video de YouTube")
                
    except Exception as e:
        print(f"Error procesando categoría: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando extracción del ABECEDARIO (A-E)...")
    
    # Mapeo de slugs del conadis (asumiendo patrón alfa-a, alfa-b)
    # Por ahora probaremos A, B, C, D, E
    letters = ['A', 'B', 'C', 'D', 'E']
    
    for letter in letters:
        slug = f"alfa-{letter.lower()}"
        scrape_category(slug, target_letters=[letter])
        
    print("\n✅ Proceso del Abecedario finalizado.")
