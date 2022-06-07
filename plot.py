import cv2
import numpy as np
from moviepy.editor import VideoFileClip

import matplotlib.image as mplimg
import matplotlib.pyplot as plt

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40 # 40
max_line_gap = 20

dir = "go Straight."


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


def draw_lines(img, lines, color=[0, 0, 255], thickness=8):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img


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
        return img

    clean_lines(left_lines, 0.4)
    clean_lines(right_lines, 0.4)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img

    # 取极值，画出一个线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])

    # cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    # cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)

    # 计算交点，控制方向
    (x1,y1),(x2,y2) = left_vtx
    (x3,y3),(x4,y4) = right_vtx
    if(x2==x1):
        k1 = 999.0
    else:
        k1=(y2-y1)*1.0/(x2-x1)
    if(x3==x4):
        k2=999.0
    else:
        k2=(y4-y3)*1.0/(x4-x3) 
    if(k1> 0 and k2 >0):
        dir = "turn LEFT."
    elif(k1< 0 and k2 <0):
        dir = "turn RIGHT."
    else:
        left_x,_ = cross_point(left_vtx,((0,1080),(1000,1080)))
        right_x,_ = cross_point(right_vtx,((0,1080),(1000,1080)))
        if(left_x>420):
            dir = "turn RIGHT: too close to borderline. "
        elif(right_x<1500):
            dir = "turn LEFT: too close to borderline. "
        else:
            point_x, point_y = cross_point(left_vtx,right_vtx)
            if(point_x>990):
                dir = "turn RIGHT. "+str(point_x)
            elif(point_x<930):
                dir = "turn LEFT. "+str(point_x)
            else:
                dir = "go Straight."

    # res_img = cv2.putText(img,dir,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    # print(dir)
    # print(point_x,point_y)

    # 直接画出所有线
    draw_lines(img,left_lines)
    draw_lines(img,right_lines)

def cross_point(line1,line2):
    (x1,y1),(x2,y2) = line1
    (x3,y3),(x4,y4) = line2
    # k1=(y2-y1)*1.0/(x2-x1)
    # k2=(y4-y3)*1.0/(x4-x3) 
    if(x2==x1):
        k1 = 999.0
    else:
        k1=(y2-y1)*1.0/(x2-x1)
    if(x3==x4):
        k2=999.0
    else:
        k2=(y4-y3)*1.0/(x4-x3) 
    b1=y1*1.0-x1*k1*1.0
    b2=y3*1.0-x3*k2*1.0
    x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0  
    return x,y

def clean_lines(lines, threshold):        
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    del_list = []

    # 斜率过滤
    for i,s in enumerate(slope):
        if abs(s) < 0.18:
            # print("删除",s)
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
    print(img.shape[0],img.shape[1])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)

    roi_edges = roi_mask(edges, roi_vtx)

    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    # res_img = cv2.putText(res_img,dir,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    
    if(flag==1):
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.savefig('../resources/gray.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(blur_gray, cmap='gray')
        plt.savefig('../resources/blur_gray.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.savefig('../resources/edges.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(roi_edges, cmap='gray')
        plt.savefig('../resources/roi_edges.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(line_img, cmap='gray')
        plt.savefig('../resources/line_img.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(res_img)
        plt.savefig('../resources/res_img.png', bbox_inches='tight')
        plt.show()

        return res_img
    else:
        return res_img

flag = 0
if(flag==1):
    # 处理图片
    img = mplimg.imread("../resources/test7.jpg")
    print(img)
    process_an_image(img)
else:
    # 处理视频
    ls = [9]
    for i in ls:
        output = '../resources/v'+str(i)+'s1.mp4'
        clip = VideoFileClip("../resources/v"+str(i)+".mp4")
        out_clip = clip.fl_image(process_an_image)
        out_clip.write_videofile(output, audio=False)


