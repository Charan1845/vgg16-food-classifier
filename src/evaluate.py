import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .data import get_datasets


def main():
    model_path = os.path.join(config.CHECKPOINT_DIR, "food101_vgg16_final.keras")
    model = tf.keras.models.load_model(model_path)

    _, test_ds, class_names = get_datasets()

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.npy")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(cm_path, cm)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
