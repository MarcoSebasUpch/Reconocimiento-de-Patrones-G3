import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json
import zipfile
import shutil

# ── Cargar modelo real (.keras) ──────────────────────────────────────────
MODEL_NAME = "modelo_EfficientNetB0_final.keras"
MODEL_CLEANED = "modelo_limpio.keras"
error_de_carga = "No se ha intentado cargar todavía"

def preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)

def desinfectar_y_cargar_modelo():
    """
    Abre el archivo .keras como un ZIP, elimina 'quantization_config' del config.json
    para evitar el bug de Keras 3 y guarda un nuevo modelo limpio para cargar de forma segura.
    """
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {MODEL_NAME}")
        
    # Crear un directorio temporal para descomprimir el modelo
    tmp_dir = "tmp_keras_model"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    try:
        # 1. Descomprimir la estructura de Keras
        with zipfile.ZipFile(MODEL_NAME, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        config_path = os.path.join(tmp_dir, "config.json")
        
        if os.path.exists(config_path):
            # 2. Leer y limpiar el JSON de configuración de forma quirúrgica
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Borramos absolutamente cualquier rastro de la propiedad 'quantization_config'
            config_content = config_content.replace('"quantization_config": null', '')
            config_content = config_content.replace('"quantization_config": {}', '')
            config_content = config_content.replace(', ,', ',')
            config_content = config_content.replace('{ ,', '{')
            config_content = config_content.replace(', }', '}')
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
                
        # 3. Volver a empaquetar todo en un nuevo archivo .keras limpio
        if os.path.exists(MODEL_CLEANED):
            os.remove(MODEL_CLEANED)
            
        with zipfile.ZipFile(MODEL_CLEANED, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, tmp_dir)
                    zip_out.write(full_path, relative_path)
                    
        # 4. Cargar el archivo modificado libre de basura interna
        custom_objects = {
            "efficientnet_preprocess": preprocess_input,
            "preprocess_input": preprocess_input
        }
        return tf.keras.models.load_model(MODEL_CLEANED, custom_objects=custom_objects, compile=False)
        
    finally:
        # Limpieza de la carpeta temporal
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

try:
    # Llamamos a nuestro desinfectante de archivos antes de iniciar Gradio
    model = desinfectar_y_cargar_modelo()
    error_de_carga = "Ninguno, el modelo cargó perfectamente en memoria."
    print("¡Modelo desinfectado y cargado exitosamente!")
# ... (Tu código actual termina aquí en el except)
except Exception as e:
    model = None
    error_de_carga = str(e)
    print(f"Error crítico al cargar el modelo: {e}")

# ──────────────────────────────────────────────────────────────────────────
#  ¡PEGA ESTO DEBAJO PARA CREAR LA INTERFAZ Y MANTENER EL SERVIDOR VIVO!
# ──────────────────────────────────────────────────────────────────────────

# ── Parámetros de la Red ──────────────────────────────────────────────────
IMG_SIZE = (224, 224)
CLASES = ["NORMAL", "PNEUMONIA"]
FINAL_THRESHOLD = 0.50

def predict_image(image):
    if image is None:
        return "Por favor, selecciona o sube una imagen.", "0.00%", "No se cargó ninguna imagen."
    
    if model is None:
        return "Error de Servidor", "0.00%", f"🔴 TensorFlow rechazó el modelo.\n\n**Razón técnica:**\n`{error_de_carga}`"

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    
    tensor = np.expand_dims(arr, axis=0)
    
    prob_neumonia = float(model.predict(tensor, verbose=0).ravel()[0])
    prob_normal = 1.0 - prob_neumonia
    
    if prob_neumonia >= FINAL_THRESHOLD:
        label = "🔴 PNEUMONIA (NEUMONÍA DETECTADA)"
        confianza = prob_neumonia
    else:
        label = "🟢 NORMAL (SIN SIGNOS DE NEUMONÍA)"
        confianza = prob_normal
        
    confianza_str = f"{confianza * 100:.2f}%"
    
    detalles_md = f"""### 📊 Parámetros de Evaluación Médica:
| Métrica | Valor Obtenido |
| :--- | :--- |
| **Arquitectura** | EfficientNetB0 (Transfer Learning) |
| **Umbral de Decisión (Threshold)** | {FINAL_THRESHOLD:.2f} |
| **Probabilidad de Neumonía** | {prob_neumonia:.6f} |
| **Probabilidad Normal** | {prob_normal:.6f} |
| **Resultado de Predicción** | **{CLASES[1] if prob_neumonia >= FINAL_THRESHOLD else CLASES[0]}** |
"""
    return label, confianza_str, detalles_md

# ── Interfaz Gráfica con Gradio Blocks ──────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align:center; margin-bottom: 15px;">
        <h1 style="color:#1a5276; font-size:2rem; margin-bottom:5px;">
            🫁 DetectAI: Clasificación de Neumonía en Radiografías de Tórax
        </h1>
        <p style="color:#5d6d7e; font-size:1rem; margin-top:0;">
            Proyecto de Reconocimiento de Patrones · Grupo 3 · EfficientNetB0 Batch 32
        </p>
    </div>
    """)
    
    gr.Markdown("### ⚠️ **Aviso:** Sistema académico de apoyo; no apto para diagnóstico clínico real.")
    
    gr.Markdown("""### Descripción del Problema
La neumonía es una infección pulmonar grave que requiere un diagnóstico rápido. Este modelo de Deep Learning 
analiza imágenes de radiografías de tórax (X-Ray) para identificar signos preliminares de la enfermedad.
""")
    
    gr.Markdown("[🔗 Ver Repositorio en GitHub](https://github.com/MarcoSebasUpch/Reconocimiento-de-Patrones---G3)")
    
    gr.HTML("<hr style='border: 0; height: 1px; background: #bbb; margin: 20px 0;'>")
    
    gr.Markdown("### 📂 Panel de Análisis")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Sube una radiografía en formato JPG/PNG")
            btn_predict = gr.Button("🔍 Ejecutar Análisis Médico", variant="primary")
            
        with gr.Column():
            output_label = gr.Textbox(label="Resultado de la Predicción", interactive=False)
            output_conf = gr.Textbox(label="Probabilidad / Nivel de Confianza", interactive=False)
            output_table = gr.Markdown(value="Esperando análisis...")
            
    btn_predict.click(
        fn=predict_image,
        inputs=input_img,
        outputs=[output_label, output_conf, output_table]
    )
    
    gr.HTML("""
    <hr style="border: 0; height: 1px; background: #bbb; margin: 20px 0;">
    <div style="text-align:center; color:#888; font-size:0.85rem;">
        DetectAI · Grupo 3 — Proyecto Universitario de Reconocimiento de Patrones
    </div>
    """)

if __name__ == "__main__":
    # En Gradio 5, el servidor se mantiene vivo automáticamente sin parámetros raros
    demo.launch(server_name="0.0.0.0", server_port=7860)
