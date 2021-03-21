import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def color_filter(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray_image=grayscale(img)
    lower_yellow = np.array([26, 43, 46], dtype = "uint8")
    upper_yellow = np.array([100, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    return mask_yw_image

def gaussian_blur(img, kernel_size=3, sigma=0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def median_blur(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size) # maybe it will be helpful in the future

def canny(img,low_threshold=100,high_threshold=200):
    # void Canny(InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false )
    # 由Sobel運算子計算梯度大小, 若是高過Gmax就是邊緣, 在Gmax, Gmin之間但與邊緣相連也是邊緣, 其餘皆不是
    return cv2.Canny(img,low_threshold,high_threshold)

def ROI(img, vertices):
    if len(img.shape) > 2: 
        channel_count = img.shape[2] # i.e. 3 or 4 depending on your image 
        ignore_mask_color = (255,) * channel_count  # if chnnel_count = 3, ignore_mask_color = (255, 255, 255)
    else: 
        ignore_mask_color = 255

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([ROI_vertices], np.int32), ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    # the left lane will have a negative slope and right positive
    def get_slope(x1,y1,x2,y2):
        return (y2-y1)/(x2-x1)
        
    global cache
    global first_frame

    y_global_min = img.shape[0]
    y_max = img.shape[0]
    l_slope, r_slope, l_lane, r_lane = [], [], [], []
    det_slope = 0.4
    alpha = 0.3

    if lines is None:
        print ('Lines is not enough!')
        return 1

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)

        y_global_min = min(y1,y2,y_global_min)

    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('No lane detected!')
        return 1
    
    l_slope_mean = np.mean(l_slope) 
    r_slope_mean = np.mean(r_slope) 
    l_mean = np.mean(np.array(l_lane),axis=0) # mean()求平均值, axis=0表示壓縮行, 求各列均值
    r_mean = np.mean(np.array(r_lane),axis=0) 

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('Dividing by zero!')
        return 1

    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0]) # 求截距
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    l_x1 = int((y_global_min - l_b)/(l_slope_mean)) 
    l_x2 = int((y_max - l_b)/(l_slope_mean))   
    r_x1 = int((y_global_min - r_b)/(r_slope_mean))
    r_x2 = int((y_max - r_b)/(r_slope_mean))

    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2])
    
    if first_frame == 1:
        next_frame = current_frame        
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)

    cache = next_frame

def weighted_img(img1, img2, alpha=0.8, beta=1.0, gamma=0.0):
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)

def hough_lines(img, rho=6, theta=np.pi/180, threshold=160, minLineLength=40, maxLineGap=25):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def CLAHE(image, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl_img = clahe.apply(image)
    return cl_img

def test_video(src):
    print('Test Lane Detection From video......')
    cap = cv2.VideoCapture(src)
    global first_frame
    first_frame = 1
    cv2.namedWindow("Lane Detection Result",cv2.WINDOW_NORMAL)

    while(1):
        ret, frame_original = cap.read()
        if not ret:
            print("Failed to open video!")
            break
        frame = frame_original
        # frame = adjust_gamma(frame_original, 0.2)
        gray_image=grayscale(frame)
        gray_image = CLAHE(gray_image)
        mask_yw_image = color_filter(frame)
        gauss_gray = gaussian_blur(mask_yw_image,kernel_size=3, sigma=0)
        canny_img = canny(gauss_gray, low_threshold=50, high_threshold=150)

        imshape = frame.shape

        height = imshape[0]
        width = imshape[1]
        '''
        region_of_interest_vertices = [ (0, height),
                                        (width / 2, height / 2),
                                        (width, height)]
        vertices = np.array([region_of_interest_vertices], dtype=np.int32)
        '''
        lower_left = [imshape[1]/9,imshape[0]]
        lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
        top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
        top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
        vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
        
        roi_image = region_of_interest(canny_img, vertices)

        line_img = hough_lines(roi_image, rho=6, theta=np.pi/180, threshold=30, minLineLength=40, maxLineGap=25)
        lane_detection_result = weighted_img(line_img, frame_original, alpha=0.8, beta=1.0, gamma=0.0)
        
        # concatenatedOutput = cv2.hconcat([frame, lane_detection_result])
        cv2.imshow("Lane Detection Result", lane_detection_result)

        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            # cv2.imwrite("result.jpg", lane_detection_result)
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_img(src):
    print('Test Lane Detection From image......')
    frame = cv2.imread(src)
    # frame = adjust_gamma(frame, 0.5)
    gray_image=grayscale(frame)
    global first_frame
    first_frame = 1

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    plt.figure(1)
    plt.imshow(hsv_img, cmap='gray')
    plt.title('HSV Image')

    lower_yellow = np.array([26, 43, 46], dtype = "uint8")
    upper_yellow = np.array([100, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    plt.figure(2)
    plt.imshow(mask_yellow, cmap='gray')
    plt.title('Yellow Mask')

    mask_white = cv2.inRange(gray_image, 200, 255)
    plt.figure(3)
    plt.imshow(mask_white, cmap='gray')
    plt.title('White Mask')

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    plt.figure(4)
    plt.imshow(mask_yw_image, cmap='gray')
    plt.title('Masked Image')

    gauss_gray = gaussian_blur(mask_yw_image,kernel_size=3, sigma=0)
    plt.figure(5)
    plt.imshow(gauss_gray, cmap='gray')
    plt.title('Gaussian Image')
    
    canny_img = canny(gauss_gray, low_threshold=50, high_threshold=150)
    plt.figure(6)
    plt.imshow(canny_img, cmap='gray')
    plt.title('Canny Edges')

    imshape = frame.shape
    '''
    # Option 1
    height = imshape[0]
    width = imshape[1]
    region_of_interest_vertices = [ (0, height),
                                        (width / 2, height / 2),
                                        (width, height)]
    vertices = np.array([region_of_interest_vertices], dtype=np.int32)
    '''
    
    # Option 2
    lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

    '''
    # Option 3
    vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    '''

    roi_image = region_of_interest(canny_img, vertices)
    plt.figure(7)
    plt.imshow(roi_image, cmap='gray')
    plt.title('ROI')

    line_img = hough_lines(roi_image, rho=6, theta=np.pi/180, threshold=40, minLineLength=60, maxLineGap=20)
    plt.figure(8)
    plt.imshow(line_img, cmap='gray')
    plt.title('Hough Lines')

    lane_detection_result = weighted_img(line_img, frame, alpha=0.8, beta=1.0, gamma=0.0)

    lane_detection_result = lane_detection_result[:,:,::-1]
    plt.figure(9)
    plt.imshow(lane_detection_result)
    plt.title('Lane Detection Result')
    plt.show()