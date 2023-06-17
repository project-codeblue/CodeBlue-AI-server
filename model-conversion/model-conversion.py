import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model = load_model('rnn_model_v3.h5')
tfjs.converters.save_keras_model(model, './predict_emergency_level_model')