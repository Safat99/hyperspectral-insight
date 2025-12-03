import tensorflow as tf
from tensorflow.keras import layers, models

def build_hyper3dnet(input_shape, n_classes, lr=1e-4):
    """
    Full Hyper3DNet architecture.
    input_shape: (win, win, bands, 1)
    """

    inp = layers.Input(shape=input_shape)

    # 3D feature encoder
    x = layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Collapse spectral-channel dimension for 2D processing
    d = input_shape[2] * 32
    x = layers.Reshape((input_shape[0], input_shape[1], d))(x)

    # Spatial encoder
    x = layers.SeparableConv2D(256, 5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model