import tensorflow as tf
from tensorflow.keras import layers, models


def build_hyper3dnet_lite(
    input_shape, 
    n_classes,
    optimizer_name: str = "adam", 
    lr: float = 1e-3,
    rho: float = 0.95,
    epsilon: float = 1e-7,
    ):
    """
    Lightweight Hyper3DNet-Lite architecture.
    
    input_shape: (win, win, bands, 1)
    optimizer_name: "adam" or "adadelta"
    lr: learning rate (used only for Adam)
    
    Optimizer parameters:
    - Adam: lr
    - Adadelta: rho, epsilon
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