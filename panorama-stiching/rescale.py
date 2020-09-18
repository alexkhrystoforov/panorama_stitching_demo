import cv2

img = cv2.imread('view1.jpg')

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)

print(img.shape)
dim = (1600, 1017)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print(img.shape)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)