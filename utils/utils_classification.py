# import os 
# import json 
# import shutil
# import numpy as np
# from PIL import Image


# def read_images_classfication(path):
#     """
#     Reads images from the specified path and labels and load it in dataset. where labels are folder names in path

#     Returns:
#     dataset
#     """

#     dataset = []

#     ## first list dir 

#     labels = os.listdir(path)

#     for label in labels:
#         label_folder = os.path.join(path, label)
#         if os.path.isdir(label_folder):
#             for img_file in os.listdir(label_folder):
#                 if img_file.endswith(".jpg") or img_file.endswith(".png"):
#                     img_path = os.path.join(label_folder, img_file)
#                     image = Image.open(img_path)
#                     dataset.append((image, label))

#     return dataset
    

import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle  # optional, for shuffling


def read_images_classification(path, image_size=(224, 224)):
    """
    Reads images from folders where each folder is a label.
    Returns X (images as numpy arrays) and y (labels).
    """

    X, y = [], []
    labels = os.listdir(path)

    for label in labels:
        label_folder = os.path.join(path, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(label_folder, img_file)
                    try:
                        image = Image.open(img_path).convert("RGB")
                        image = image.resize(image_size)
                        X.append(np.array(image))
                        y.append(label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Shuffle
    X, y = shuffle(X, y, random_state=42)

    return X, y
