# Imports
#from main import save_img
#from main import get_terminal_width
from operations import total_loss
from operations import get_vgg_layers
from operations import calculate_gram_matrix
from tools import save_img
from tools import get_terminal_width

# Import libraries
import tensorflow as tf
from keras import applications
from tqdm.autonotebook import tqdm
import os
import shutil

# Parse flags
#FLAGS = flags.FLAGS

# Style and content layers
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
CONTENT_LAYERS = ['block5_conv2']
NUM_STYLE_LAYERS = len(STYLE_LAYERS)

# Model parameters
epochs = 5 # was 20
steps_per_epoch = 100  # was 500
tv_weight = 1e8
content_weight = 10000
style_weight = 0.10
learning_rate = 0.02
beta_1 = 0.99
beta_2 = 0.999
epsilon = 0.10


# Optimizer
# OPT = tf.optimizers.Adam(learning_rate = FLAGS.learning_rate, beta_1 = FLAGS.beta_1, beta_2 = FLAGS.beta_2, epsilon = FLAGS.epsilon)

# Call model
def call_model(inputs, vgg):
    inputs = inputs * 255.0
    preprocessed_input = applications.vgg19.preprocess_input(inputs)
    outputs = vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:NUM_STYLE_LAYERS], outputs[NUM_STYLE_LAYERS:])

    style_outputs = [calculate_gram_matrix(style_output) for style_output in style_outputs]
    content = {content_name: value for content_name, value in zip(CONTENT_LAYERS, content_outputs)}
    style = {style_name: value for style_name, value in zip(STYLE_LAYERS, style_outputs)}
    return {'content': content, 'style': style}

# Train model
def train_model(content_img, style_img):
    img = tf.Variable(content_img)
    vgg = get_vgg_layers(STYLE_LAYERS + CONTENT_LAYERS)
    vgg.trainable = False
    style_targets = call_model(style_img, vgg)['style']
    content_targets = call_model(content_img, vgg)['content']

    for n in range(epochs):
        for _ in tqdm(
                    iterable=range(steps_per_epoch),
                    ncols=int(get_terminal_width() * .9),
                    desc=tqdm.write(f'Epoch {n + 1}/{epochs}'),
                    unit=' steps',
            ):
        
            img = train_step(img, vgg, style_targets, content_targets)
            #img = tf.Variable(img) # ff
        save_img(img, n + 1)

# Training steps
def train_step(img, vgg, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = call_model(img, vgg)
        loss = total_loss(outputs, content_targets, style_targets, CONTENT_LAYERS, STYLE_LAYERS, img)

    grad = tape.gradient(loss, img)

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)
    opt.apply_gradients([(grad, img)])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img