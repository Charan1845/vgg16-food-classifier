# VGG16 Food Classifier

Transfer-learning food classifier built on VGG16, as a learning project before a larger application.

## Architecture

VGG16 convolutional base (ImageNet-pretrained) -> custom head:

- Blocks 1-4: frozen
- Block 5: fine-tuned in stage 2
- GlobalAveragePooling2D
- Dense(256, sigmoid)
- Dropout(0.5)
- Dense(101, softmax) — classes limited to `config.NUM_CLASSES` for faster iteration

Trained on a subset of the [Food-101](https://www.tensorflow.org/datasets/catalog/food101) dataset.

## Usage

```bash
pip install -r requirements.txt
python -m src.train       # two-stage training
python -m src.evaluate    # test-set metrics
python -m src.predict path/to/image.jpg
```
