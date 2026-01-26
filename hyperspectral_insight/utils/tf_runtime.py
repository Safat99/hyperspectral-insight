def configure_tensorflow_runtime(
    disable_xla: bool = True,
    enable_memory_growth: bool = True,
):
    import os
    import tensorflow as tf

    if disable_xla:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
        os.environ["TF_ENABLE_XLA"] = "0"

    if enable_memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Must be set before GPUs are initialized
                pass
