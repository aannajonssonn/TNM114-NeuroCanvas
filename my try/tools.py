
import os
import shutil

from tensorflow import io
import tensorflow as tf

max_dim = 512

# Where to save the generated images
output_dir ='generated/'

# Function to save image
def save_img(img, epoch):
    file_name = f'{epoch}.jpg'
    output_path = os.path.join(output_dir, file_name)
    for i in img:
        img = tf.image.encode_jpeg(tf.cast(i * 255, tf.uint8), format='rgb')
        tf.io.write_file(output_path, img)

# Function to load image
def load_img(img_path):
    img = io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape, antialias=True)
    img = img[tf.newaxis, :]
    return img

# Function to get terminal width
def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width