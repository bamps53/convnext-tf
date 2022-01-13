# ConvNeXt Tensorflow

This is unofficial tensorflow keras implementation of ConvNeXt.  
Its based on official [PyTorch implementation](https://github.com/facebookresearch/ConvNeXt).

## Pre-trained Models

| name | resolution | pretrain | acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| `convnext_tiny_224` | 224x224 |ImageNet-1K | 82.1 | 28M | 4.5G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| `convnext_small_224` | 224x224 |ImageNet-1K | 83.1 | 50M | 8.7G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| `convnext_base_224` | 224x224 |ImageNet-21K-1K | 85.8 | 89M | 15.4G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| `convnext_base_384` | 384x384 |ImageNet-21K-1K | 86.8 | 89M | 45.0G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| `convnext_large_224` | 224x224 |ImageNet-21K-1K | 86.6 | 198M | 34.4G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| `convnext_large_384` | 384x384 |ImageNet-21K-1K | 87.5 | 198M | 101.0G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |
| `convnext_xlarge_224` | 224x224 |ImageNet-21K-1K | 87.0 | 350M | 60.9G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| `convnext_xlarge_384` | 384x384 |ImageNet-21K-1K | 87.8 | 350M | 179.0G | [github](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |

## Note
I've ported only ImageNet-21K-1K weights for base, large and xlarge models.  
If you want to convert another pretrained weight in official repo, you can refer to [this script](https://github.com/bamps53/convnext-tf/blob/master/convert_weights.py) or just let me know.

## Examples
```python
import tensorflow as tf
from models.convnext_tf import create_model

x = tf.zeros((1, 224, 224, 3), dtype=tf.float32)

model = create_model('convnext_tiny_224', input_shape=(224, 224), pretrained=True)
out = model(x) # (1, 1000)

model = create_model('convnext_tiny_224', input_shape=(224, 224), num_classes=1, pretrained=True)
out = model(x) # (1, 1)

model = create_model('convnext_tiny_224', input_shape=(224, 224), include_top=False, pretrained=True)
out = model(x) # (1, 16, 16, 768)
```

## Reference
https://github.com/facebookresearch/ConvNeXt  
https://github.com/rishigami/Swin-Transformer-TF  