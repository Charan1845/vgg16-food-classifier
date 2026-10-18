import tensorflow as tf
import tensorflow_datasets as tfds

from . import config


def _preprocess(image, label, training):
    image = tf.image.resize(image, config.IMAGE_SIZE)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label


def _make_split(split, training):
    ds, info = tfds.load(
        "food101",
        split=split,
        with_info=True,
        as_supervised=True,
        shuffle_files=training,
    )
    if config.NUM_CLASSES < info.features["label"].num_classes:
        ds = ds.filter(lambda img, lbl: lbl < config.NUM_CLASSES)
    ds = ds.map(
        lambda img, lbl: _preprocess(img, lbl, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.shuffle(1000, seed=config.SEED)
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, info


def get_datasets():
    """Food-101 only ships train/validation splits; validation doubles as the held-out test set."""
    train_ds, info = _make_split("train", training=True)
    test_ds, _ = _make_split("validation", training=False)
    class_names = info.features["label"].names[: config.NUM_CLASSES]
    return train_ds, test_ds, class_names
