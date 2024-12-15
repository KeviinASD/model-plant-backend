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
TFLITE_MODEL_PATH  = os.path.join(settings.BASE_DIR, "predictions/models/modelo_entrenado_20.tflite")

if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {TFLITE_MODEL_PATH}")
else:
    print(f"Modelo cargado de: {TFLITE_MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@csrf_exempt
def predict(request):
    if request.method == "POST":
        if "image" not in request.FILES:
            return JsonResponse({"error": "Archivo no encontrado en la solicitud"}, status=400)

        try:
            # Procesar la imagen
            image = request.FILES["image"]
            img = Image.open(image)
            img = img.convert("RGB")
            img = img.resize((512, 512))  # Ajustar tamaño según el modelo
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Realizar la predicción con TensorFlow Lite
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            # Mapear la predicción a clases
            classes = ["Healthy", "Multiple diseases", "Rust", "Scab"]
            predicted_class = classes[np.argmax(predictions)]

            return JsonResponse({"prediction": predicted_class, "probabilities": predictions.tolist()})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Método no permitido"}, status=405)