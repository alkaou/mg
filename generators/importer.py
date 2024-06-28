# Importing Libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf, keras
from tensorflow.keras import layers, models, optimizers
import numpy as np
from tensorflow.keras.saving import register_keras_serializable