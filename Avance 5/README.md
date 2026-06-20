# Clasificación de Neumonía en Radiografías de Tórax

Este proyecto implementa un sistema de clasificación binaria de radiografías de tórax para distinguir entre las clases NORMAL y PNEUMONIA. Se evaluaron modelos de transferencia de aprendizaje basados en EfficientNetB0, MobileNetV2 y DenseNet121.

## 1. Requisitos

Para ejecutar el proyecto, instalar las dependencias con:

pip install -r requirements.txt

## 2. Evaluación
Para cada modelo se calcula la probabilidad de pertenecer a la clase PNEUMONIA. Luego, usando un umbral de decisión, se obtiene la clase final:

```python
threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)
```

## ADVERTENCIA
En caso de usar algún modelo, recordar que fueron guardados en formato '.keras'. 
Como algunos modelos incluyen una capa `Lambda` asociada a la función `preprocess_input`, es necesario importar la función de preprocesamiento correspondiente antes de cargar cada modelo.

```python
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
```

Luego, al cargar el modelo se debe pasar asignar la función de preprocesamiento al 'custom_objects' correspondiente:

```python
for model_path in model_paths:
    model_name = Path(model_path).stem
    print("\nEvaluando:", model_name)

    lower_path = model_path.lower()

    if "efficientnet" in lower_path:
        custom_objects = {"preprocess_input": efficientnet_preprocess}

    elif "mobilenet" in lower_path:
        custom_objects = {"preprocess_input": mobilenet_preprocess}

    elif "densenet" in lower_path:
        custom_objects = {"preprocess_input": densenet_preprocess}

```
