import tensorflow as tf
from tensorflow.keras import layers, models

def build_full3dcnn(
    input_shape,
    n_classes,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    rho: float = 0.95,
    epsilon: float = 1e-7,
):
    """
    Full 3D CNN from attention-based band selection paper.

    input_shape: (H, W, B, 1)

    Optimizer parameters:
    - Adam: lr
    - Adadelta: rho, epsilon
    """

    inp = layers.Input(shape=input_shape)

    # --------- Feature Extractor ---------
    x = layers.Conv3D(
        32, kernel_size=(3, 3, 3),
        padding="same", activation="relu"
    )(inp)

    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    x = layers.Conv3D(
        64, kernel_size=(3, 3, 3),
        padding="same", activation="relu"
    )(x)

    # --------- Global pooling ---------
    x = layers.GlobalAveragePooling3D()(x)

    # --------- Classifier ---------
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)

    # -------- Optimizer selection (TUNABLE) --------
    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr
        )

    elif optimizer_name.lower() == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(
            rho=rho,
            epsilon=epsilon
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model