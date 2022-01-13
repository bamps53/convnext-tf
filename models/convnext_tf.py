import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Layer,
                                     LayerNormalization)


def drop_path(inputs, drop_prob, is_training):
    # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class Block(Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, prefix=''):
        super().__init__()
        self.dwconv = DepthwiseConv2D(
            kernel_size=7, padding='same')  # depthwise conv
        self.norm = LayerNormalization(epsilon=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = Dense(4 * dim)
        self.act = tf.keras.activations.gelu
        self.pwconv2 = Dense(dim)
        self.drop_path = DropPath(drop_path)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value
        self.prefix = prefix

    def build(self, input_shape):
        self.gamma = tf.Variable(
            initial_value=self.layer_scale_init_value * tf.ones((self.dim)),
            trainable=True,
            name=f'{self.prefix}/gamma')
        self.built = True

    def call(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(tf.keras.Model):
    r""" ConvNeXt
        A Tensorflow keras impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        include_top (bool): whether to add head or just use it as feature extractor. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], include_top=True,
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.include_top = include_top
        self.downsample_layers = []  # stem and 3 intermediate downsampling conv layers
        stem = tf.keras.Sequential([
            Conv2D(dims[0], kernel_size=4, strides=4, padding='same'),
            LayerNormalization(epsilon=1e-6)]
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = tf.keras.Sequential([
                LayerNormalization(epsilon=1e-6),
                Conv2D(dims[i+1], kernel_size=2, strides=2, padding='same')]
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = []  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = tf.keras.Sequential(
                [Block(dim=dims[i], drop_path=dp_rates[cur + j],
                       layer_scale_init_value=layer_scale_init_value, prefix=f'block{i}') for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        if self.include_top:
            self.avg = GlobalAveragePooling2D()
            self.norm = LayerNormalization(epsilon=1e-6)  # final norm layer
            self.head = Dense(num_classes)
        else:
            self.avg = None
            self.norm = None
            self.head = None

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.avg(x)
            x = self.norm(x)
            x = self.head(x)
        return x


model_urls = {
    "convnext_tiny_224": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_tiny_1k_224_ema.h5",
    "convnext_small_224": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_small_1k_224_ema.h5",
    "convnext_base_224": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_base_22k_1k_224.h5",
    "convnext_base_384": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_base_22k_1k_384.h5",
    "convnext_large_224": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_large_22k_1k_224.h5",
    "convnext_large_384": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_large_22k_1k_384.h5",
    "convnext_xlarge_224": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_xlarge_22k_1k_224_ema.h5",
    "convnext_xlarge_384": "https://github.com/bamps53/convnext-tf/releases/download/v0.1/convnext_xlarge_22k_1k_384_ema.h5",
}


model_configs = dict(
    convnext_tiny=dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ),
    convnext_small=dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768]
    ),
    convnext_base=dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024]
    ),
    convnext_large=dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536]
    ),
    convnext_xlarge=dict(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048]
    ),
)


def create_model(model_name='convnext_tiny_1k', input_shape=(224, 224), num_classes=1000, include_top=True, pretrained=True, use_tpu=False, **kwargs):
    cfg = model_configs['_'.join(model_name.split('_')[:2])]
    net = ConvNeXt(num_classes, cfg['depths'],
                   cfg['dims'], include_top, **kwargs)
    net(tf.keras.Input(shape=(*input_shape, 3)))
    if pretrained is True:
        url = model_urls[model_name]
        pretrained_ckpt = tf.keras.utils.get_file(
            f'{model_name}.h5', url, untar=False)
        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost')
            net.load_weights(
                pretrained_ckpt, options=load_locally, skip_mismatch=True)
        else:
            net.load_weights(pretrained_ckpt, skip_mismatch=True, by_name=True)

    return net
