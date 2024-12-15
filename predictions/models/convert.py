import tensorflow as tf

# Ruta del modelo original
MODEL_PATH = "modelo_entrenado_20.h5"
TFLITE_MODEL_PATH = "modelo_entrenado_20.tflite"

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Modelo convertido guardado en: {TFLITE_MODEL_PATH}")
