from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet
from happywhale.util import TARGET_SHAPE

"""emb_size = 64

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(emb_size, activation='sigmoid')
])

embedding_model.summary()
"""
