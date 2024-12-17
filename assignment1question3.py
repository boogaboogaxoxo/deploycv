import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(name, n, m, i, title):
    plt.subplot(n, m, i + 1)
    plt.imshow(name, cmap='gray')
    plt.title(title)
    plt.axis('off')
img1 = cv2.imread('cat.png')
img2 = cv2.imread('dog.png')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

low_pass = cv2.GaussianBlur(img1_rgb, (21, 21), 0)
low_pass_inter = cv2.GaussianBlur(img2_rgb, (21, 21), 0)
high_pass = img2_rgb - low_pass_inter
high_pass = np.clip(high_pass, 0, 255)

merged_img = np.clip(low_pass_inter + high_pass, 0, 255)
images = [img1_rgb, img2_rgb, low_pass, low_pass_inter, high_pass, merged_img]
titles = ["cat","dog", "cat'nt","uh","dog'nt","cog"]
for i in range(len(images)):
    show(images[i],2,3,i,titles[i])
plt.tight_layout()
plt.show()


