from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

model.summary()