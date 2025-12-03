from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

def build_shallow_cnn(input_shape, n_classes, lr=1e-4):
    """
    Simple 2D CNN used as a baseline.
    Input shape: (patch_h, patch_w, bands)
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
