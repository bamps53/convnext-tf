import os

import cv2
import numpy as np
import tensorflow as tf
import timm
import torch
from models.convnext import *  # just to register model
from models.convnext_tf import create_model

model_urls = [
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
]


def load_test_image(image_size=(224, 224)):
    from skimage.data import chelsea
    img = chelsea()  # Chelsea the cat
    img = cv2.resize(img, image_size)
    img = img / 255
    img = (img - np.array([0.485, 0.456, 0.406])) / \
        np.array([0.229, 0.224, 0.225])
    return img


def load_from_url(model, url):
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True)
    model.load_state_dict(checkpoint["model"])
    return model


def get_pt_state_dict(pt_model):
    pt_state_dict = {}
    for n, p in pt_model.named_parameters():
        pt_state_dict[n] = p.detach().numpy()
    return pt_state_dict


def get_tf_state_dict(tf_model):
    tf_state_dict = {}
    for layer in tf_model.layers:
        for var in layer.weights:
            if ('dense' in var.name) and ('kernel' in var.name):
                w = tf.transpose(var, [1, 0]).numpy()
            elif ('depthwise_conv2d' in var.name) and ('kernel' in var.name):
                w = tf.transpose(var, [2, 3, 0, 1]).numpy()
            elif ('conv2d' in var.name) and ('kernel' in var.name):
                w = tf.transpose(var, [3, 2, 0, 1]).numpy()
            else:
                w = var.numpy()
            tf_state_dict[var.name] = w
    return tf_state_dict


def convert_weights(tf_model, pt_state_dict, key_map):
    for layer in tf_model.layers:
        for var in layer.weights:
            pt_key = key_map[var.name]
            pt_w = pt_state_dict[pt_key]
            if ('dense' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [1, 0])
            elif ('depthwise_conv2d' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [2, 3, 0, 1])
            elif ('conv2d' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [2, 3, 1, 0])
            else:
                tf_w = pt_w
            var.assign(tf_w)


def test_models(pt_model, tf_model):
    img = load_test_image()
    pt_x = torch.tensor(img[None, ]).permute(0, 3, 1, 2).to(torch.float32)
    with torch.no_grad():
        pt_y = torch.softmax(pt_model(pt_x), -1).numpy()

    tf_x = tf.convert_to_tensor(img[None, ])
    tf_y = tf.nn.softmax(tf_model(tf_x)).numpy()
    np.testing.assert_allclose(pt_y, tf_y, rtol=1e-5)


def main():
    os.makedirs('weights', exist_ok=True)
    for model_url in model_urls:
        file_name = os.path.basename(model_url)
        model_name = '_'.join(file_name.split('_')[:2])
        save_name = file_name.replace("pth", "h5")
        if os.path.exists(f'weights/{save_name}'):
            continue

        pt_model = timm.create_model(model_name)
        pt_model = load_from_url(pt_model, model_url)
        pt_model.eval()
        pt_state_dict = get_pt_state_dict(pt_model)

        tf_model = create_model(model_name)
        tf_state_dict = get_tf_state_dict(tf_model)

        key_map = {t: p for t, p in zip(
            tf_state_dict.keys(), pt_state_dict.keys())}
        convert_weights(tf_model, pt_state_dict, key_map)

        test_models(pt_model, tf_model)
        print(f'successfully converted {model_name}!')

        tf_model.save_weights(f'weights/{save_name}')


if __name__ == '__main__':
    main()
