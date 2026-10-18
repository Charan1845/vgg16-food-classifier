import os

import tensorflow as tf

from . import config
from .data import get_datasets
from .model import build_model, set_fine_tune


def _callbacks(name):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{name}_best.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds, class_names = get_datasets()
    model, base = build_model()

    # Stage 1: train the new head only, conv base frozen.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.STAGE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.STAGE1_EPOCHS,
        callbacks=_callbacks("stage1"),
    )

    # Stage 2: unfreeze block5, fine-tune at a low learning rate.
    set_fine_tune(base, config.FINE_TUNE_BLOCK)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.STAGE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.STAGE2_EPOCHS,
        callbacks=_callbacks("stage2"),
    )

    final_path = os.path.join(config.CHECKPOINT_DIR, "food101_vgg16_final.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
