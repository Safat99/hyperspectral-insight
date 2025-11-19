import tensorflow as tf
from tensorflow.keras import layers, models

def build_hybridsn(input_shape, n_classes):
    """
    Hybrid spectral-spatial model (HybridSN).
    """
    inp = layers.Input(shape=input_shape)

    # 3D CNN spectral+spatial
    x = layers.Conv3D(8, (3,3,7), activation="relu")(inp)
    x = layers.Conv3D(16, (3,3,5), activation="relu")(x)
    x = layers.Conv3D(32, (3,3,3), activation="relu")(x)

    _, H, W, D, C = x.shape
    x = layers.Reshape((H, W, D*C))(x)

    # 2D refinement
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model