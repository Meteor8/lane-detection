import cv2
import numpy as np

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40 # 40
max_line_gap = 20

dir = "mid"


def roi_mask(img, vertices):
    roi_car = np.array([[(750,560),(350,960),(1570,960),(1170,560)]])
    roi_und = np.array([[(160,820),(160,920),(1760,920),(1760,820)]])


    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)
    cv2.fillConvexPoly(mask, roi_car, (0,0,0))
    cv2.fillConvexPoly(mask, roi_und, (0,0,0))

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    left_lane,right_lane = draw_lanes(line_img, lines)
    return left_lane,right_lane


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1<960 and x2<960:
                left_lines.append(line)
            elif x1>960 and x2>960:
                right_lines.append(line)
            # k = (y2 - y1) / (x2 - x1)
            # if k < 0:
            #     left_lines.append(line)
            # else:
            #     right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return None,None

    clean_lines(left_lines, 0.4)
    clean_lines(right_lines, 0.4)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return None,None

    # 取极值，画出一个线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])

    return left_vtx,right_vtx


def calc_dir(left_lane, right_lane):
    if(left_lane != None and right_lane != None):
        # 计算交点，控制方向
        (x1,y1),(x2,y2) = left_lane
        (x3,y3),(x4,y4) = right_lane
        if(x1==x2):
            k1 = 999.0
        else:
            k1=(y2-y1)*1.0/(x2-x1)
        if(x3==x4):
            k2 = 999.0
        else:
            k2=(y4-y3)*1.0/(x4-x3) 
        if(k1> 0 and k2 >0):
            dir = "left"
        elif(k1< 0 and k2 <0):
            dir = "right"
        else:
            left_x,_ = cross_point(left_lane,((0,1080),(1000,1080)))
            right_x,_ = cross_point(right_lane,((0,1080),(1000,1080)))
            if(left_x>450):
                dir = "right"
            elif(right_x<1470):
                dir = "left"
            else:
                point_x, _ = cross_point(left_lane,right_lane)
                if(point_x>990):
                    dir = "right"
                elif(point_x<930):
                    dir = "left"
                else:
                    dir = "mid"
    else:
        dir = "mid"
    return dir


def cross_point(line1,line2):
    (x1,y1),(x2,y2) = line1
    (x3,y3),(x4,y4) = line2
    k1=(y2-y1)*1.0/(x2-x1)
    k2=(y4-y3)*1.0/(x4-x3) 
    b1=y1*1.0-x1*k1*1.0
    b2=y3*1.0-x3*k2*1.0
    x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0  
    return x,y

def clean_lines(lines, threshold):
    slope = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (x2==x1):
                slope.append(999.0)
            else:
                slope.append((y2 - y1) / (x2 - x1))
    # 斜率过滤
    del_list = []
    for i,s in enumerate(slope):
        if abs(s) < 0.18:
            del_list.append(i)
    for i in del_list[::-1]:
        slope.pop(i)
        lines.pop(i)

    # 异常值过滤
    # while len(lines) > 0:
    #     mean = np.mean(slope)
        
    #     diff = [abs(s - mean) for s in slope]
    #     print(diff)
    #     idx = np.argmax(diff) # ???
    #     if diff[idx] > threshold:
    #         slope.pop(idx)
    #         lines.pop(idx)
    #     else:
    #         break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def process_an_image(img):
    roi_vtx = np.array([[(160, 490), (160, 920), (1760, 920), (1760, 490)]])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)

    roi_edges = roi_mask(edges, roi_vtx)

    left_lane,right_lane = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)

    dir = calc_dir(left_lane, right_lane)
    return dir


def detection(img):
    img = np.array(img)
    dir = process_an_image(img)
    return dir
