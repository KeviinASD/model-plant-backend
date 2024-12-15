import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np
import io
import json
from django.conf import settings
import os

# Create your views here.
# Cargar el modelo previamente entrenado
MODEL_PATH = os.path.join(settings.BASE_DIR, "predictions/models/modelo_entrenado_20.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {MODEL_PATH}")
else:
    print(f"Modelo cargado de: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


from io import BytesIO

@csrf_exempt
def predict(request):
    if request.method == "POST":
        print("Archivos recibidos:", request.FILES)  # Imprime los archivos recibidos
        if "image" not in request.FILES:
            return JsonResponse({"error": "Archivo no encontrado en la solicitud"}, status=400)

        try:
            # Leer el archivo como un flujo de bytes
            image = request.FILES["image"]
            image_bytes = image.read()
            img = Image.open(BytesIO(image_bytes))  # Usar BytesIO para abrir la imagen
            img = img.convert("RGB")  # Asegurar formato RGB
            img = img.resize((512, 512))  # Ajustar tamaño según el modelo

            # Procesar la imagen
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Realizar la predicción
            predictions = model.predict(img_array)[0]
            classes = ["Healthy", "Multiple diseases", "Rust", "Scab"]
            predicted_class = classes[np.argmax(predictions)]

            return JsonResponse({"prediction": predicted_class, "probabilities": predictions.tolist()})
        except Exception as e:
            print("Error en el procesamiento de la imagen:", e)  # Imprime el error en la consola
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Método no permitido"}, status=405)
