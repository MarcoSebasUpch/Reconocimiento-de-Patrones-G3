# ── PARCHES DE COMPATIBILIDAD CRÍTICOS (PYTHON 3.13 & HF HUB MODERNO) ─────
import sys
import types

# 1. Parche para simular 'audioop' (eliminado en Python 3.13)
if 'audioop' not in sys.modules:
    mock_audioop = types.ModuleType('audioop')
    mock_audioop.error = Exception
    sys.modules['audioop'] = mock_audioop
    print("¡Parche audioop inyectado con éxito!")

# 2. Parche para simular 'HfFolder' (eliminado en versiones modernas de huggingface_hub)
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'HfFolder'):
        class MockHfFolder:
            @staticmethod
            def get_token(): return None
            @staticmethod
            def save_token(token): pass
            @staticmethod
            def delete_token(): pass
        huggingface_hub.HfFolder = MockHfFolder
        sys.modules['huggingface_hub'].HfFolder = MockHfFolder
        print("¡Parche HfFolder inyectado con éxito!")
except Exception as e:
    print(f"No se pudo aplicar el parche de HF Hub: {e}")

# ── INICIALIZACIÓN NORMAL DEL SISTEMA ─────────────────────────────────────
import os
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import zipfile
import shutil

MODEL_NAME = "modelo_EfficientNetB0_final.keras"
MODEL_CLEANED = "modelo_limpio.keras"
error_de_carga = "No se ha intentado cargar todavía"

def preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)

# ── Desinfección Estructural del Modelo Keras ──────────────────────────────
def desinfectar_y_cargar_modelo():
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(f"No se encontró el archivo del modelo original: {MODEL_NAME}")
        
    tmp_dir = "tmp_keras_model"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    try:
        with zipfile.ZipFile(MODEL_NAME, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        config_path = os.path.join(tmp_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            config_content = config_content.replace('"quantization_config": null', '')
            config_content = config_content.replace('"quantization_config": {}', '')
            config_content = config_content.replace(', ,', ',')
            config_content = config_content.replace('{ ,', '{')
            config_content = config_content.replace(', }', '}')
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
                
        if os.path.exists(MODEL_CLEANED):
            os.remove(MODEL_CLEANED)
            
        with zipfile.ZipFile(MODEL_CLEANED, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, tmp_dir)
                    zip_out.write(full_path, relative_path)
                    
        custom_objects = {
            "efficientnet_preprocess": preprocess_input,
            "preprocess_input": preprocess_input
        }
        return tf.keras.models.load_model(MODEL_CLEANED, custom_objects=custom_objects, compile=False)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

# ── Carga de la Red Neuronal en Memoria ────────────────────────────────────
try:
    model = desinfectar_y_cargar_modelo()
    error_de_carga = "Ninguno."
    print("¡Modelo desinfectado y cargado exitosamente!")
except Exception as e:
    model = None
    error_de_carga = str(e)
    print(f"Error crítico al cargar el modelo: {e}")

# ── Lógica de Inferencia Médica ───────────────────────────────────────────
IMG_SIZE = (224, 224)
FINAL_THRESHOLD = 0.50

def predict_image(image):
    if image is None:
        return "Por favor, sube una imagen.", "0.00%"
    if model is None:
        return "Error de Servidor", f"Modelo no disponible: {error_de_carga}"

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    tensor = np.expand_dims(arr, axis=0)
    
    prob_neumonia = float(model.predict(tensor, verbose=0).ravel()[0])
    prob_normal = 1.0 - prob_neumonia
    
    if prob_neumonia >= FINAL_THRESHOLD:
        label = "🔴 PNEUMONIA (DETECTADA)"
        confianza = f"{prob_neumonia * 100:.2f}%"
    else:
        label = "🟢 NORMAL (SIN SIGNOS)"
        confianza = f"{prob_normal * 100:.2f}%"
        
    return label, confianza

# ── Maquetación de Interfaz Gráfica ─────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🫁 DETECCIÓN DE NEUMONÍA EN RADIOGRAFÍAS DE TÓRAX")
    gr.Markdown("### Proyecto de Reconocimiento de Patrones · Grupo 3")
    gr.Markdown("⚠️ **Aviso:** Sistema académico de apoyo; no apto para diagnóstico clínico real.")
    gr.Markdown("""
    ### Descripción del Problema
    La neumonía es una infección pulmonar grave que requiere un diagnóstico rápido. Este modelo de Deep Learning 
    analiza imágenes de radiografías de tórax (X-Ray) para identificar signos preliminares de la enfermedad, 
    buscando optimizar los tiempos de triaje médico. No se cuentan con suficientes médicos especialistas
    que realicen el diagnóstico y este número se reduce al interior del país.
    """)
    gr.Markdown("[🔗 Ver Repositorio en GitHub](https://github.com/MarcoSebasUpch/Reconocimiento-de-Patrones---G3)")
    
    gr.Markdown("""
    ---
    📊 **Estado actual del proyecto:** Se eligio la arquitectura de EfficientNet ya que consiguio mejores parámetros,
    especialmente de sensibilidad (menor cantidad de falsos positivos)
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            # Forzamos type="pil" y removemos dependencias avanzadas de la API
            input_img = gr.Image(label="Sube tu Radiografía (X-Ray)", type="pil")
            btn_predict = gr.Button("🔍 Ejecutar Análisis", variant="primary")
            
        with gr.Column():
            output_label = gr.Textbox(label="Resultado de la Predicción")
            output_conf = gr.Textbox(label="Nivel de Confianza")
            
    btn_predict.click(
        fn=predict_image,
        inputs=[input_img],
        outputs=[output_label, output_conf],
        api_name=False # TRUCO MAESTRO: Apaga la API para esta ruta, evitando el bucle del error
    )

if __name__ == "__main__":
    # Desactivamos por completo todas las propiedades de API externas del backend
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        show_api=False,
        max_threads=4
    )