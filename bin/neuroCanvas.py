# NeuroCanvas
# Authors: Anna Jonsson & Daniel Wärulf
# TNM114 - AI for interactive Media, Linköpings University 2023
# Date: 2023-10-05

# Imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import time
import functools
import urllib.request

#%tensorflow_version 2.x
import tensorflow as tf

import keras.utils as kp_image # import image_dataset_from_directory 
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import IPython.display

# Enable eager execution
#tf.enable_eager_execution()
#print("Eager execution: {}".format(tf.executing_eagerly()))

# Figure size and axis grid for mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

# Global values (content and style)
content_path = "content/Green_Sea_Turtle_grazing_seagrass.jpg"
style_path = "style/The_Great_Wave_off_Kanagawa.jpg" 

# Visualize the input
def load_img(path_to_img):
    max_dim = 512  
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.LANCZOS)
    img = np.array(img)     # Copilot suggested np.array(img)
    return img

# Display the image
def imshow(img, title=None):
    # Remove the batch dimension if it exists
    if img.ndim == 4:
        img = np.squeeze(img, axis=0)

    # Normalize for display
    img = img.astype('uint8')
    plt.imshow(img)

    if title is not None:
        plt.title(title)

   # plt.show()

# Show content and style images
plt.figure(figsize=(10,10))
content = load_img(content_path).astype('uint8')    # Converts the numpy array to int array
style = load_img(style_path).astype('uint8')
plt.subplot(1, 2, 1)
imshow(content, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()

# Preprocess the images
def preprocess_img(img):
    img = load_img(img).astype('uint8') 
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# De-process the images
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
    
    # Inverse of preprocessing
    x[:, :, 0] += 103.939   # Mean for blue channel
    x[:, :, 1] += 116.779   # Mean for green channel
    x[:, :, 2] += 123.68    # Mean for red channel
    x = x[:,:,::-1]         # 
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Intermediate layers using VGG19
# These intermediate layers are necessary to define the representation of content and style from our images.
# Intermediate layers represent higher and higher order features: from edges (early layers) to abstract features (later layers)

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Build the model
# load VGG19, feed input tensor to the model, extract feature maps of the content, style, and genereated images
# VGG19 is more simple than resnet, inception mfl, making the feature maps more suitable for style transfer

def get_model():
    # Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    # Build model
    return models.Model(vgg.input, model_outputs)

# Content loss
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Style loss
def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]

    # Gram matrix
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target)) #/ (4. * (channels ** 2) * (width * height) ** 2)

# Run gradient descent
def get_feature_representations(model, content_path, style_path):
    # Load our images in
    #content_image = load_img(content_path)
    #style_image = load_img(style_path)

    # Preprocess image
    content_image = preprocess_img(content_path)
    style_image = preprocess_img(style_path)

    # Batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer for style_layer in style_outputs[:num_style_layers]] # style_layer[0]
    content_features = [content_layer for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# Compute loss and gradients
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and style representations at our desired layers
    model_outputs = model(init_image) # Callable since we're using eager execution
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight
    # total_variation_weight = total_variation_weight * total_variation_loss(init_image)

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score

# Compute gradients
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    
    # Ensure all_loss has three values: loss, style_score, and content_score
    total_loss = all_loss[0]
    gradients = tape.gradient(total_loss, cfg['init_image'])
    return gradients, all_loss

# Optimization loop
def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    # We don't want to train the layers of the model
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = preprocess_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1) # TODO: Is there any other optimizer we can use? What happens when we change values?

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        # Compute gradients
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss # unpacks all elements of all_loss into loss, style_score, content_score
        opt.apply_gradients([(grads, init_image)]) # opt = optimizer, backpropagation to minimize loss?
        # Clipping gradients
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

    if loss < best_loss:
        # Update best loss and best image from total loss
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

    if i % display_interval == 0:
        start_time = time.time()

        # Use the .numpy() method to get the concrete numpy array
        plot_img = init_image.numpy()
        plot_img = deprocess_img(plot_img)
        imgs.append(plot_img)
        IPython.display.clear_output(wait=True)
        IPython.display.display_png(Image.fromarray(plot_img))
        print('Iteration: {}'.format(i))
        print('Total loss: {:.4e}, '
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss

# Run the style transfer
best, best_loss = run_style_transfer(content_path, style_path, num_iterations=1000)
im = Image.fromarray(best)

# Extract image names from paths
content_name = os.path.splitext(os.path.basename(content_path))[0]
style_name = os.path.splitext(os.path.basename(style_path))[0]

# Construct a dynamic filename based on content and style image names and current timestamp
timestamp = time.strftime("%Y%m%d%H%M%S")
filename = f"generated_image_{content_name}_{style_name}_{timestamp}.png"

# Save the image
im.save(filename)

# Visualize output
def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()

# Show results
show_results(im, content_path, style_path)