import tensorflow as tf
import tensorflow_datasets as tfds

from . import config


def _label_lookup(all_class_names):
    """Maps each original Food-101 label id to its index in config.SELECTED_CLASSES, or -1."""
    new_index = {name: i for i, name in enumerate(config.SELECTED_CLASSES)}
    table = [new_index.get(name, -1) for name in all_class_names]
    return tf.constant(table, dtype=tf.int64)


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
    lookup = _label_lookup(info.features["label"].names)
    ds = ds.map(lambda img, lbl: (img, tf.gather(lookup, lbl)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda img, lbl: lbl >= 0)
    ds = ds.map(
        lambda img, lbl: _preprocess(img, lbl, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.shuffle(1000, seed=config.SEED)
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets():
    """True 3-way split: 90% of `train` for gradient updates, 10% held back for
    early stopping / checkpoint selection. The `validation` split is never
    touched here, so it stays a clean, unseen test set (see get_test_dataset)."""
    train_ds = _make_split("train[:90%]", training=True)
    val_ds = _make_split("train[90%:]", training=False)
    return train_ds, val_ds, config.SELECTED_CLASSES


def get_test_dataset():
    """The untouched Food-101 `validation` split, used only for final reporting."""
    test_ds = _make_split("validation", training=False)
    return test_ds, config.SELECTED_CLASSES
