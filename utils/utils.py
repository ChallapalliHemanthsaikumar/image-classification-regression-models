import os 

import random 
import matplotlib.pyplot as plt


def show_images(images, labels, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        idx = random.randint(0, len(images) - 1)
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx].permute(1, 2, 0))
        plt.title(labels[idx])
        plt.axis("off")
    plt.show()