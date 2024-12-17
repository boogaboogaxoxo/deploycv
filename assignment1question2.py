import cv2
import numpy as np
import matplotlib.pyplot as plt
def show(name,n,m,i,title):
        plt.subplot(n, m, i + 1)
        plt.imshow(name, cmap='gray')
        plt.title(title)
        plt.axis('off')

cap = cv2.VideoCapture(0)
checker, frame = cap.read()
cap.release()

if checker:
    cv2.imshow('uwu', frame)
else:
    print('where image')

grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_,bin_thresh = cv2.threshold(grey,128,255,cv2.THRESH_BINARY)
grey_thresh = (grey//16)*16
grad_x = cv2.Sobel(grey,cv2.CV_64F,1,0,ksize=3)
grad_y = cv2.Sobel(grey,cv2.CV_64F,0,1,ksize=3)
mag = cv2.magnitude(grad_x,grad_y)
img_sobel = cv2.convertScaleAbs(mag)
img_canny = cv2.Canny(grey,100,200)
img_gauss = cv2.GaussianBlur(grey,(5,5),0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img_sharp = cv2.filter2D(grey,-1,kernel)
img_bgr=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

images = [grey,bin_thresh,grey_thresh,img_sobel,img_canny,img_gauss,img_sharp,img_bgr]
titles = ["Grayscale", "B&W Threshold", "16-Level Gray", "Sobel", "Canny",
          "Gaussian Blur", "Sharpened", "RGB to BGR"]
for i in range(len(images)):
    show(images[i],2,4,i,titles[i])
plt.tight_layout()
plt.show()

