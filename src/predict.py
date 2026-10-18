import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import config


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model_path = os.path.join(config.CHECKPOINT_DIR, "food101_vgg16_final.keras")
    model = tf.keras.models.load_model(model_path)
    class_names = tfds.builder("food101").info.features["label"].names[: config.NUM_CLASSES]

    image = load_image(args.image_path)
    preds = model.predict(image, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][: args.top_k]
    for idx in top_indices:
        print(f"{class_names[idx]}: {preds[idx]:.4f}")


if __name__ == "__main__":
    main()
