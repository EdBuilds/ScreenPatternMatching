import cv2
import numpy as np

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

img_rgb=cv2.imread('Bilge.png')
img_gray= cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template=cv2.imread('Bilge_sprite.png',-1)
template= cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
leftUpCorner=(87,45)
rightBottCorner=(356,584)
tilexWidth=(rightBottCorner[0]-leftUpCorner[0])/6+1
tileyWidth=(rightBottCorner[1]-leftUpCorner[1])/12+1
print(tilexWidth)
print(tileyWidth)
for xIndex in range(6):
    x=leftUpCorner[0]+xIndex*tilexWidth
    for yIndex in range(12):
        y=leftUpCorner[1]+yIndex*tileyWidth
        tileImg=img_gray[y:y+tileyWidth,x:x+tilexWidth] 
        if mse(tileImg,template)>4000:
            cv2.imshow('Detected',tileImg)
            cv2.waitKey(0)
img_rgb = img_rgb[leftUpCorner[1]:rightBottCorner[1], leftUpCorner[0]:rightBottCorner[0]]
res=cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
w, h = template.shape[::-1]
threshold=0.7
loc = np.where(res>=threshold)
ret,thresh1 = cv2.threshold(res,0.5,1,cv2.THRESH_BINARY)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
