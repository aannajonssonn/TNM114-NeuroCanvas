# Imports
import tensorflow as tf
from tensorflow import linalg
from keras import models
from keras.applications.vgg19 import VGG19
#from absl import flags

# Parse flags
#FLAGS = flags.FLAGS

# Model parameters
tv_weight = 1e8
content_weight = 10000
style_weight = 0.10

# Get VGG Layers
def get_vgg_layers(layer_names):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = models.Model([vgg.input], outputs)
    return model

# Gram Matrix
def calculate_gram_matrix(tensor):
    input_shape = tf.shape(tensor)
    result = linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Compute Loss
def compute_loss(outputs, targets):
    return tf.add_n([
        tf.reduce_mean(tf.square(outputs[name] - targets[name]))
        for name in outputs.keys()
    ])

# High Frequencies
def get_high_frequencies(img):
    x = img[:, :, 1:, :] - img[:, :, :-1, :]
    y = img[:, 1:, :, :] - img[:, :-1, :, :]
    return x, y

# Variation Loss
def variation_loss(img):
    x, y = get_high_frequencies(img)
    return tf.reduce_mean(tf.square(x)) + tf.reduce_mean(tf.square(y))

# Total Loss
def total_loss(outputs, content_targets, style_targets, content_layers, style_layers, img):
    content_loss = compute_loss(outputs['content'], content_targets)
    style_loss = compute_loss(outputs['style'], style_targets)

    content_loss *= content_weight / len(content_layers)
    style_loss *= style_weight / len(style_layers)

    total_loss = style_loss + content_loss
    total_loss += tv_weight * variation_loss(img)

    return total_loss
