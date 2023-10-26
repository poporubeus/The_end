import os
import numpy as np
from PIL import Image

def load_galaxy_data(num_train, num_test, rng):
    train_folderY = '/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/galaxy/'
    test_folderY = '/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/galaxy/'
    train_folderN = '/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/no_galaxy/'
    test_folderN = '/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/no_galaxy/'


    nogalaxy_train = [(i.replace(".jpg", "")) for i in os.listdir(train_folderN) if i != ".DS_Store"]
    yesgalaxy_train = [(i.replace(".jpg", "")) for i in os.listdir(train_folderY) if i != ".DS_Store"]
    nogalaxy_test = [(i.replace(".jpg", "")) for i in os.listdir(test_folderN) if i != ".DS_Store"]
    yesgalaxy_test = [(i.replace(".jpg", "")) for i in os.listdir(test_folderY) if i != ".DS_Store"]

    train_imagesY = []
    train_imagesN = []
    train_labelsY = []
    train_labelsN = []

    test_imagesY = []
    test_imagesN = []
    test_labelsY= []
    test_labelsN = []

    # Load training images and labels
    train_classesY = sorted(os.listdir(yesgalaxy_train))
    
    for class_index, class_name in enumerate(train_classesY):
        class_folderY = os.path.join(yesgalaxy_train, class_name)
        for image_file in os.listdir(class_folderY):
            image_path = os.path.join(class_folderY, image_file)
            image = Image.open(image_path)
            image = np.array(image)

            train_imagesY.append(image)
            train_labelsY.append(class_index)

    train_classesN = sorted(os.listdir(nogalaxy_train))
    for class_index, class_name in enumerate(train_classesN):
        class_folderN = os.path.join(train_folderN, class_name)
        for image_file in os.listdir(class_folderN):
            image_path = os.path.join(class_folderN, image_file)
            image = Image.open(image_path)
            image = np.array(image)

            train_imagesN.append(image)
            train_labelsN.append(class_index)

    # Load testing images and labels
    test_classesY = sorted(os.listdir(yesgalaxy_test))
    for class_index, class_name in enumerate(test_classesY):
        class_folderY_TEST = os.path.join(test_folderY, class_name)
        for image_file in os.listdir(class_folderY_TEST):
            image_path = os.path.join(class_folderY_TEST, image_file)
            image = Image.open(image_path)
            image = np.array(image)

            test_imagesY.append(image)
            test_labelsY.append(class_index)
    test_classesN = sorted(os.listdir(nogalaxy_test))
    for class_index, class_name in enumerate(test_classesN):
        class_folderN_TEST = os.path.join(test_folderN, class_name)
        for image_file in os.listdir(class_folderN_TEST):
            image_path = os.path.join(class_folderN_TEST, image_file)
            image = Image.open(image_path)
            image = np.array(image)

            test_imagesN.append(image)
            test_labelsN.append(class_index)
    
    # Convert lists to numpy arrays
    train_images = np.array(train_imagesY + train_imagesN)
    train_labels = np.array(train_labelsY + train_labelsN)
    test_images = np.array(test_imagesY + test_imagesN)
    test_labels = np.array(test_labelsY + test_labelsN)

    # Normalize data (optional)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Subsample train and test split
    train_indices = rng.choice(len(train_labels), num_train, replace=False)
    test_indices = rng.choice(len(test_labels), num_test, replace=False)

    x_train, y_train = train_images[train_indices], train_labels[train_indices]
    x_test, y_test = test_images[test_indices], test_labels[test_indices]

    return x_train, y_train, x_test, y_test


import numpy as np
from numpy.random import default_rng

# Set the number of training and testing samples you want
num_train = 30
num_test = 80

# Set the random number generator seed for reproducibility
rng = default_rng(seed=42)


x_train, y_train, x_test, y_test = load_galaxy_data(num_train, num_test, rng)

# Check the shape of the loaded data
print("Training data shape:", x_train.shape)  # (num_train, height, width, channels)
print("Training labels shape:", y_train.shape)  # (num_train,)
print("Testing data shape:", x_test.shape)  # (num_test, height, width, channels)
print("Testing labels shape:", y_test.shape)  # (num_test,)