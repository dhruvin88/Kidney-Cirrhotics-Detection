from PIL import Image
import numpy as np
import pydicom
import cv2
import os

def imgTransfrom(fileName,normal):
    img = pydicom.dcmread(fileName)
    
    imgArray = img.pixel_array
    
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
    
    #outline = out[:,:,1] - out[:,:,0] == 0
    #out[outline  == False] = 0
    #out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) 

    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.fastNlMeansDenoising(out,None,10,7,21)

    outline = out[:,:,1] - out[:,:,0] == 0
    #Canny edge detection
    if normal:
        out = cv2.Canny(out,50,150)
    else:
        out = cv2.Canny(out,50,50)
    out[outline  == False] = 0
    
    #write out the image
    fileName = os.path.basename(fileName)
    fileName,_ = os.path.splitext(fileName)
    cv2.imwrite("./Output_Images/"+fileName+".pgm", out)
    
def resize(path):
    for item in os.listdir(path):
        if os.path.isfile(path+item):
            if not item.startswith('.'):
                im = Image.open(path+item)
                f, e = os.path.splitext(path+item)
                imResize = im.resize((128,128), Image.ANTIALIAS)
                imResize.save(path+item)

####
normals = [f for f in os.listdir('./Normals10') if not f.startswith('.')]
cirrhotics = [f for f in os.listdir("./Cirrhotics10") if not f.startswith('.')]

for x in normals:
    files = [f for f in os.listdir('./Normals10/'+x) if not f.startswith('.')]
    for y in files:
        imgTransfrom("./Normals10/"+x+"/"+y,True)

for x in cirrhotics:
    files = [f for f in os.listdir('./Cirrhotics10/'+x) if not f.startswith('.')]
    for y in files:
        imgTransfrom("./Cirrhotics10/"+x+"/"+y,False)
        
##resize all images
resize("./Output_Images/")
