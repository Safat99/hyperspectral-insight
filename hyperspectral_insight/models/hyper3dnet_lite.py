import tensorflow as tf
from tensorflow.keras import layers, models


def build_hyper3dnet_lite(input_shape, n_classes, lr=1e-4):
    """
    Lightweight Hyper3DNet-Lite architecture.
    input_shape: (win, win, bands, 1)
    """
    inp = layers.Input(shape=input_shape)

    # 3D extractor
    x = layers.Conv3D(16, (3,3,3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(16, (3,3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    s = input_shape[2] * 16
    x = layers.Reshape((input_shape[0], input_shape[1], s))(x)

    # 2D encoder
    x = layers.SeparableConv2D(320, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(256, 3, padding="same", activation="relu")(x)
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