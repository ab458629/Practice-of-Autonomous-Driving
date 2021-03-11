'''
P46091204 蔡承穎
1. 程式名稱：lane_detect.py
2. 程式內容：Lane Detection
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size=3, sigma=0.25):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def canny(img,low_threshold=100,high_threshold=200):
    # void Canny(InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false )
    return cv2.Canny(img,low_threshold,high_threshold)

def draw_lines(lines, im, edgesIm):
    imshape = im.shape
    # 分成正斜率以及負斜率
    slopePositiveLines = [] # the list contains [x1, y1, x2, y2, slope]
    slopeNegativeLines = []
    yValues = []
    
    addedPos = False
    addedNeg = False
    for currentLine in lines:   
        # Get points of current Line
        for x1,y1,x2,y2 in currentLine:
            lineLength = ((x2-x1)**2 + (y2-y1)**2)**.5 
            if lineLength > 30: # length of line need to be long enough 
                if x2 != x1:    # avoid to divide by zero
                    slope = (y2-y1)/(x2-x1) 
                    # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                    if slope > 0: 
                        tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta)
                        ang = np.arctan(tanTheta)*180/np.pi # obtain the value from radius to degree
                        if abs(ang) < 85 and abs(ang) > 20:
                            slopeNegativeLines.append([x1,y1,x2,y2,-slope]) # add positive slope line
                            yValues.append(y1)
                            yValues.append(y2)
                            addedPos = True # note that we added a positive slope line

                    if slope < 0:
                        tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                        ang = np.arctan(tanTheta)*180/np.pi
                        if abs(ang) < 85 and abs(ang) > 20:
                            slopePositiveLines.append([x1,y1,x2,y2,-slope]) # add negative slope line
                            yValues.append(y1)
                            yValues.append(y2)
                            addedNeg = True # note that we added a negative slope line
           
            
    # If we didn't get any positive lines, go though again and just add any positive slope lines         
    if not addedPos:
        for currentLine in lines:
            for x1,y1,x2,y2 in currentLine:
                slope = (y2-y1)/(x2-x1)
                if slope > 0:
                    # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                    tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                    ang = np.arctan(tanTheta)*180/np.pi
                    if abs(ang) < 80 and abs(ang) > 15:
                        slopeNegativeLines.append([x1,y1,x2,y2,-slope])
                        yValues.append(y1)
                        yValues.append(y2)
    
    # If we didn't get any negative lines, go through again and just add any negative slope lines
    if not addedNeg:
        for currentLine in lines:
            for x1,y1,x2,y2 in currentLine:
                slope = (y2-y1)/(x2-x1)
                if slope < 0:
                    # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                    tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                    ang = np.arctan(tanTheta)*180/np.pi
                    if abs(ang) < 85 and abs(ang) > 15:
                        slopePositiveLines.append([x1,y1,x2,y2,-slope])           
                        yValues.append(y1)
                        yValues.append(y2)
                   
    
    if not addedPos or not addedNeg:
        print('Not enough lines found!')
    
    #　To get the mean of Pos/Neg Slope
    positiveSlopes = np.asarray(slopePositiveLines)[:,4]
    posSlopeMedian = np.mean(positiveSlopes)
    posSlopeStdDev = np.std(positiveSlopes)
    posSlopesGood = []
    for slope in positiveSlopes:
       if abs(slope-posSlopeMedian) < 0.9: # if abs(slope-posSlopeMedian) < posSlopeMedian*.2:
            posSlopesGood.append(slope)
    posSlopeMean = np.mean(np.asarray(posSlopesGood))
            
    
    negativeSlopes = np.asarray(slopeNegativeLines)[:,4]
    negSlopeMedian = np.mean(negativeSlopes)
    negSlopeStdDev = np.std(negativeSlopes)
    negSlopesGood = []
    for slope in negativeSlopes:
        if abs(slope-negSlopeMedian) < 0.9:
            negSlopesGood.append(slope)
    negSlopeMean = np.mean(np.asarray(negSlopesGood))
        
    xInterceptPos = []
    for line in slopePositiveLines:
            x1 = line[0]
            y1 = im.shape[0]-line[1] # y axis is flipped => 變成我們習慣的y軸向上
            slope = line[4]
            if slope is not 0:
                yIntercept = y1-slope*x1
                xIntercept = -yIntercept/slope # find x intercept based off y = mx+b
                xInterceptPos.append(xIntercept) # add x intercept
            
    xIntPosMed = np.median(xInterceptPos) # get median 
    xIntPosGood = [] # if not near median we get rid of that x point
    for line in slopePositiveLines:
            x1 = line[0]
            y1 = im.shape[0]-line[1]
            slope = line[4]
            yIntercept = y1-slope*x1
            xIntercept = -yIntercept/slope
            if abs(xIntercept-xIntPosMed) < 0.35*xIntPosMed: # check if near median
                xIntPosGood.append(xIntercept)
                    
    xInterceptPosMean = np.mean(np.asarray(xIntPosGood)) # average of good x intercepts for positive line
    
    # Negative Lines 
    xInterceptNeg = []
    for line in slopeNegativeLines:
        x1 = line[0]
        y1 = im.shape[0]-line[1]
        slope = line[4]
        if slope is not 0:
            yIntercept = y1-slope*x1
            xIntercept = -yIntercept/slope
            xInterceptNeg.append(xIntercept)
                
    xIntNegMed = np.median(xInterceptNeg)
    xIntNegGood = []
    for line in slopeNegativeLines:
        x1 = line[0]
        y1 = im.shape[0]-line[1]
        slope = line[4]
        yIntercept = y1-slope*x1
        xIntercept = -yIntercept/slope
        if abs(xIntercept-xIntNegMed)< 0.35*xIntNegMed: 
                xIntNegGood.append(xIntercept)
                
    xInterceptNegMean = np.mean(np.asarray(xIntNegGood))
    
    laneLines = np.zeros_like(edgesIm)   # make new black image
    colorLines = im.copy()

    try:
        # Positive Slope Line
        slope = posSlopeMean
        x1 = xInterceptPosMean
        y1 = 0
        y2 = imshape[0] - (imshape[0]-imshape[0]*0.35)
        x2 = (y2-y1)/slope + x1

        # Plot positive slope line
        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
        cv2.line(laneLines,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(255,255,0),2) # plot line yellow
        cv2.line(colorLines,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(0,0,255),4) # plot line on color image

        # Negative Slope Line
        slope = negSlopeMean
        x1N = xInterceptNegMean
        y1N = 0
        x2N = (y2-y1N)/slope + x1N

        # Plot negative Slope Line
        x1N = int(round(x1N))
        x2N = int(round(x2N))
        y1N = int(round(y1N))
        cv2.line(laneLines,(x1N,imshape[0]-y1N),(x2N,imshape[0]-y2),(255,255,0),2)
        cv2.line(colorLines,(x1N,im.shape[0]-y1N),(x2N,imshape[0]-y2),(0,0,255),4) # plot line on color iamge

        def blend_img(im):
            laneFill = im.copy()
            vertices = np.array([[(x1,im.shape[0]-y1),(x2,im.shape[0]-y2),  (x2N,imshape[0]-y2),
                                                (x1N,imshape[0]-y1N)]], dtype=np.int32)
            color = [30,255,0]
            cv2.fillPoly(laneFill, vertices, color)
            opacity = .25
            blendedIm =cv2.addWeighted(laneFill,opacity,im,1-opacity,0,im)
            cv2.line(blendedIm,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(0,0,255),4) # plot line on color image
            cv2.line(blendedIm,(x1N,im.shape[0]-y1N),(x2N,imshape[0]-y2),(0,0,255),4) # plot line on color image
            return blendedIm

        blendedIm = blend_img(im)

        return laneLines, colorLines, blendedIm

    except:
        print("There are some NaN number when draw line!")

   
def weighted_img(img1, img2, alpha=0.8, beta=1.0, gamma=0.0):
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)

def hough_lines(masked_img, rho=6, theta=np.pi/180, threshold=160, minLineLength=40, maxLineGap=25):
    lines = cv2.HoughLinesP(masked_img, 2, np.pi/180, 45, np.array([]), 40, 100)
    allLines = np.zeros_like(masked_img)
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(allLines,(x1,y1),(x2,y2),(255,255,0),2)
    return lines, allLines

def test_video(src):
    cap = cv2.VideoCapture(src)
    ret, img = cap.read()
    imshape = img.shape
    print(imshape)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (imshape[1],  imshape[0]))

    while(1):
        ret, img = cap.read()
        if not ret:
            print("Failed to open video!")
            break

        gray_img=grayscale(img)
        smooth_img = gaussian_blur(gray_img, kernel_size=3, sigma=0)
        canny_img = canny(smooth_img, low_threshold=60, high_threshold=150)
        vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
        mask = np.zeros_like(canny_img)
        color = 255
        cv2.fillPoly(mask, vertices, color)
        masked_img = cv2.bitwise_and(canny_img, mask)
        lines, allLines = hough_lines(masked_img)
        laneLines, colorLines, blendedIm = draw_lines(lines, img, canny_img)

        out.write(blendedIm)
        cv2.imshow("Lane Detection Result", blendedIm)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def test_img(src):
    print("Test Lane Detection From img......")
    img = mpimg.imread(src)
    img_name = src
    imshape = img.shape
    plt.figure(1)
    plt.imshow(img)
    plt.title(img_name)

    gray_img=grayscale(img)
    plt.figure(2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscaled image')

    smooth_img = gaussian_blur(gray_img, kernel_size=3, sigma=0)
    plt.figure(3)
    plt.imshow(smooth_img, cmap='gray')
    plt.title('Smoothed image')

    canny_img = canny(smooth_img, low_threshold=60, high_threshold=150)
    plt.figure(4)
    plt.imshow(canny_img, cmap='gray')
    plt.title('Edge Detection')

    vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    mask = np.zeros_like(canny_img)
    color = 255
    cv2.fillPoly(mask, vertices, color)
    plt.figure(5)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

    masked_img = cv2.bitwise_and(canny_img, mask)
    plt.figure(6)
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Image')

    lines, allLines = hough_lines(masked_img)
    plt.figure(7)
    plt.imshow(allLines,cmap='gray')
    plt.title('All Hough Lines Found')

    laneLines, colorLines, blendedIm = draw_lines(lines, img, canny_img)
    plt.figure(8)
    plt.imshow(laneLines,cmap='gray')
    plt.title('Lane Lines')

    plt.figure(9)
    plt.imshow(colorLines)
    plt.title('Lane Lines Color Image')

    plt.figure(10)
    plt.imshow(blendedIm)
    plt.title('Lane Detection Result')
    plt.show()

if __name__ == '__main__': 

    if len(sys.argv) < 2:
        print('No Argument!')
        sys.exit()

    if (sys.argv[1] == 'test_video'):
        print('Test Lane Detection From video......')
        test_video(sys.argv[2])

    if (sys.argv[1] == 'test_img'):
        test_img(sys.argv[2])

