import skimage
from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize two constants
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

# Multiply
result = tf.multiply(x1, x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()


# Loads in the data from the training test files
def load_data(data_directory):
    # For each directory in the data directory passed if directory is indeed a directory
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    # Empty lists to hold labels and images
    labels = []
    images = []

    # For each directory, each directory represents a label
    for d in directories:
        label_directory = os.path.join(data_directory, d)

        # Saves each file name if the filename ends with .ppm (Is an image file)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]

        # For each file read the image and append it to the dictionary d
        for f in file_names:
            # Appends the image list with the image data read in by skimage

            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

# Plots the amount of images for a given label
def plot_labels(labels):
    # Get the unique labels
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images[labels.index(label)]
        # Define 64 subplots
        plt.subplot(8, 8, i)
        # Don't include axes
        plt.axis('off')
        # Add a title to each subplot
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        # Add 1 to the counter
        i += 1
        # And you plot this first image
        plt.imshow(image)

    # Show the plot
    plt.show()

def plot_random_images(images):
    # Determine the (random) indexes of the images that you want to see
    traffic_signs = [300, 2250, 3650, 4000]

    # Fill out the subplots with the random images that you defined
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]], cmap='gray') # cmap is specified as gray as it usually uses a thermal heatmap
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))

# Resizes all the images to be 28x28 pixels
# Using the skimage transform library
def resize_images(images):
    # Rescale the images in the `images` array
    images28 = [transform.resize(image, (28, 28)) for image in images]

    return images28

def colour_to_gray(images28):
    # Convert `images28` to an array as rgb2gray expects an array
    images28 = np.array(images28)

    # Convert `images28` to grayscale
    grayImages28 = rgb2gray(images28)

    return grayImages28



ROOT_PATH = "C:\\Users\\Eoinw\\Dropbox\\College\\Year 3\\Programming for Data Analytics\\Labs\\Tensor\\BelgiumSigns"
train_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_directory)

# plot_labels(labels)
# plot_random_images(images)
images28 = resize_images(images)
grayImages28 = colour_to_gray(images28)

plot_random_images(grayImages28)

