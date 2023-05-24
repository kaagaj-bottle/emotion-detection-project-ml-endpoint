import tensorflow as tf
import numpy as np


model=tf.keras.models.load_model("model.h5")

def model_output(input):
    result_prob=model.predict(input)
    return np.argmax(result_prob[0])
