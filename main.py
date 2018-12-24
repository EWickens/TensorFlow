import skimage
from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import random

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
        plt.imshow(images[traffic_signs[i]],
                   cmap='gray')  # cmap is specified as gray as it usually uses a thermal heatmap
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


def tensor_init(images28):
    # A tensor in this case is basically an image

    # Initialize placeholders AKA Uninitialized variables
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # True array

    # Flatten the input data into an array of type [None, 784] instead of [None,28,28]
    # From my understanding it takes a three dimensional array and makes it into a two
    # dimension array

    images_flat = tf.contrib.layers.flatten(x)

    # Logits are outputs of the neural network before going through softmax function
    # Which normalises the output?

    # Fully connected layer [None ,62] ( Since there are 62 different road signs )
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Define a loss function (How far a model is from the correct result, tries to minimize loss)
    # Cross Entropy - Measures how similar how two distributions are
    # As cross entropy decreases P(truth) and q (predicted result) are more similar
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))
    # Define an optimizer - Optimizes the output using the given loss function
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)
    print("accuracy: ", accuracy)

    tf.set_random_seed(1234)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for i in range(201):  # According to guide 201 is chosen as you want to be able to register the last loss value?
        print('EPOCH', i)  # An Epoch is one full training cycle on the training set
        _, accuracy_val = sess.run([train_op, accuracy],
                                   feed_dict={x: images28, y: labels})  # Feeds the data into the neural net
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

        # Pick 10 random images
    sample_indexes = random.sample(range(len(images28)), 10)
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i], cmap="gray")

    plt.show()

    testImages, testLabels = load_data(test_data_directory)
    testImages28 = resize_images(testImages)
    grayTestImages28 = colour_to_gray(testImages28)

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: grayTestImages28})[0]

    # Calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(testLabels, predicted)])

    # Calculate the accuracy
    accuracy = match_count / len(testLabels)

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))


ROOT_PATH = "C:\\Users\\Eoinw\\Dropbox\\College\\Year 3\\Programming for Data Analytics\\Labs\\Tensor\\BelgiumSigns"
train_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_directory)

# plot_labels(labels)
# plot_random_images(images)
images28 = resize_images(images)
grayImages28 = colour_to_gray(images28)

# plot_random_images(grayImages28)



tensor_init(grayImages28)


