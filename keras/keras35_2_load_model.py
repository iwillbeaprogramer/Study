from tensorflow.keras.models import load_model
model = load_model('../data/h5/save_keras35.h5')

model.summary()