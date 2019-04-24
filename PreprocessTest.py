import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2
import os

filePath = "./Cirrhotics10/c07/c07-pvthin1 - 8225.dcm"
#filePath = "Normals10/n10/n10-nethick1 - 8210.dcm"

img = pydicom.dcmread(filePath)
imgArray = img.pixel_array
plt.imshow(imgArray, cmap=plt.cm.bone)
plt.show()
    
# Create outline based on the green outline
outline = imgArray[:,:,1] - imgArray[:,:,0]
    
# Create mask where white is what we want, black otherwise
(_, contours, _) = cv2.findContours(outline,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(outline) 
cv2.drawContours(mask, contours, 0, 255, -1)
    
# Extract out the object and place into output image
out = np.zeros_like(imgArray)
out[mask == 255] = imgArray[mask == 255]
    
# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
out = out[topx:bottomx+1, topy:bottomy+1] 
plt.imshow(out, cmap=plt.cm.bone)
plt.show()

#Gaussian Filter and Denoising
out = cv2.GaussianBlur(out, (5, 5), 0)
out = cv2.fastNlMeansDenoising(out,None,10,7,21)
plt.imshow(out, cmap=plt.cm.bone)
plt.show() 

outline = out[:,:,1] - out[:,:,0] == 0
#out[outline  == False] = 0

#Canny edge detection
out = cv2.Canny(out,50,150)
out[outline  == False] = 0

plt.imshow(out, cmap=plt.cm.bone)
plt.show()

out = cv2.resize(out, (128, 128))
fileName = os.path.basename(filePath)
fileName,_ = os.path.splitext(fileName)
cv2.imwrite("./Output_Images/"+fileName+".pgm", out)
 
