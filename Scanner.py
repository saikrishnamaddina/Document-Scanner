import cv2
import numpy as np
import rect
import sys

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

image = cv2.imread(sys.argv[1])

# resize image so it can be processed
# choose optimal dimensions such that important content is not lost
image = cv2.resize(image, (1500, 880))

# creating copy of original image
orig = image.copy()

# convert to grayscale and blur to smooth
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#blurred = cv2.medianBlur(gray, 5)

# apply Canny Edge Detection
edged = cv2.Canny(blurred, 0, 50)
orig_edged = edged.copy()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
im1,contours,hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]


# get approximate contour
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)
    area = cv2.contourArea(c)
    print(area)
    if len(approx) == 4:
        target = approx
        break


# mapping target points to 800x800 quadrilateral
approx = rectify(target)
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

M = cv2.getPerspectiveTransform(approx,pts2)
dst = cv2.warpPerspective(orig,M,(800,800))

cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


# using thresholding on warped image to get scanned effect (If Required)
ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.namedWindow("Original.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Original.jpg", orig)
cv2.namedWindow("Original Gray.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Original Gray.jpg", gray)
cv2.namedWindow("Original Blurred.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Original Blurred.jpg", blurred)
cv2.namedWindow("Original Edged.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Original Edged.jpg", orig_edged)
cv2.namedWindow("Outline.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Outline.jpg", image)
cv2.namedWindow("Thresh Binary.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Thresh Binary.jpg", th1)
cv2.namedWindow("OThresh mean.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Thresh mean.jpg", th2)
cv2.namedWindow("Thresh gauss.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Thresh gauss.jpg", th3)
cv2.namedWindow("Otsu's.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("Otsu's.jpg", th4)
cv2.namedWindow("dst.jpg",cv2.WINDOW_NORMAL)
cv2.imshow("dst.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
