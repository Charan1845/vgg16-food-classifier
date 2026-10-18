from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

from . import config


def build_model():
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMAGE_SIZE + (3,),
    )
    base.trainable = False

    inputs = layers.Input(shape=config.IMAGE_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config.HEAD_DENSE_UNITS, activation=config.HEAD_DENSE_ACTIVATION)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


def set_fine_tune(base, block_name=config.FINE_TUNE_BLOCK):
    """Unfreeze only the conv layers belonging to `block_name` (e.g. 'block5')."""
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(block_name)
