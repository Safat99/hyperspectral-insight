import tensorflow as tf
from tensorflow.keras import layers, models

def build_full3dcnn(input_shape, n_classes):
    """
    attention based band seletion paper's cnn network

    Args:
        input_shape : (H, W, B, 1)
    """
    inp = layers.Input(shape=input_shape)
    
     # --------- Feature Extractor ---------
    x = layers.Conv3D(32, kernel_size=(3,3,3), padding="same", activation="relu")(inp)
    x = layers.MaxPool3D(pool_size=(2,2,2))(x)
    x = layers.Conv3D(64, kernel_size=(3,3,3), padding="same", activation="relu")(x)
    
    x = layers.GlobalAveragePooling3D(keepdims=True)(x)
    
    x = layers.Reshape((64,))(x) # Flatten output → shape becomes (batch, 64)
    
    # --------- Classifier ---------
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    
    model = models.Model(inp, out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy", # important for one-hot labels
        metrics=["accuracy"]
    )
    
    return model