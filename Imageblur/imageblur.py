from tkinter import W
import cv2
import numpy as np
import os
from bezier import point2d,fit
from start_point import line
import pandas as pd
import math
from scipy import ndimage
import argparse
import cut

img_path = "image/"
new_img_path = "new_image/"
out_img_path = "blur/"
cp_img_path = "cp_image/"
ep_img_path = "ep_image/"
res_img_path = "blur_image"
new_img_test = "new_test_image/"
out_img_test = "blur_test/"
img_list = os.listdir(img_path)
v = 251
w = 1280
h = 720

path_test_data = "data/"
path_imu = "/gyro.csv"
sep_time = 80000
gap = 5
gap_w = w
gap_h = h
d = 1
f_x = 380*2
f_y = 380/2*3
m_x = w/2
m_y = h/2
sep_num = 3

data_start_point = []
for i in range(0,w+gap_w,gap_w):
    for j in range(0,h+gap_h,gap_h):
        data_start_point.append((j,i))

n_c = 2
data_h_point = []
for i in range(n_c):
    for j in range(n_c):
        data_h_point.append((j*h/(n_c-1),i*w/(n_c-1)))

def nearest_sampler(image, coords):
    """
    Nearest Neighbour sampler

    Parameters
    ----------
    image: ndarray
        source image. shape is [height, width, channels]
    coords: ndarray
        coordinates to be interpolated, the length of last axis should be 2,
        meaning 2D coordinate

    Returns
    -------
    output: ndarray
        the interpolated image, same shape as coords except the last axis
    """
    height, width, channels = image.shape[0:3]
    output_shape = list(coords.shape)
    coords = np.reshape(coords, (-1, output_shape[-1]))
    output_shape[-1] = channels
    coords = np.round(coords).astype(np.int32)
    idx = (coords[:, 0] >= 0) & (coords[:, 0] < width) & \
          (coords[:, 1] >= 0) & (coords[:, 1] < height)
    output = np.zeros((coords.shape[0], channels), dtype=np.uint8)
    output[idx] = image[coords[idx, 1], coords[idx, 0], :]
    output = np.reshape(output, output_shape)
    return output


def perspective_transform_matrix(uv, xy):
    """
    Compute perspective transform matrix

    Parameters
    ----------
    uv: ndarray
        coordinates of feature points in original image. shape is [n, 2], n is
        the number of points
    xy: ndarray
        coordinates of feature points in perspective transformed image. shape
        is same as uv

    Returns
    -------
    transform_matrix: ndarray
        transform matrix. shape is [3, 3]
    """
    A = np.zeros((2 * xy.shape[0], 8))
    B = np.zeros((2 * xy.shape[0], 1))
    for i in range(xy.shape[0]):
        A[2 * i, 0] = xy[i, 0]
        A[2 * i, 1] = xy[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 3] = 0.0
        A[2 * i, 4] = 0.0
        A[2 * i, 5] = 0.0
        A[2 * i, 6] = -uv[i, 0] * xy[i, 0]
        A[2 * i, 7] = -uv[i, 0] * xy[i, 1]

        A[2 * i + 1, 0] = 0.0
        A[2 * i + 1, 1] = 0.0
        A[2 * i + 1, 2] = 0.0
        A[2 * i + 1, 3] = xy[i, 0]
        A[2 * i + 1, 4] = xy[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -uv[i, 1] * xy[i, 0]
        A[2 * i + 1, 7] = -uv[i, 1] * xy[i, 1]

        B[2 * i] = uv[i, 0]
        B[2 * i + 1] = uv[i, 1]
    T = np.linalg.inv(A.T @ A) @ A.T @ B
    transform_matrix = np.append(T, [1.0]).reshape([3, 3]).T
    return transform_matrix


def perspective_coordinates(transform_matrix, height, width):
    """
    Compute perspective coordinates acoording to the transform matrix and the
    transformed image shape

    Parameters
    ----------
    transform_matrix: ndarray
        perspective transform matrix. shape is [3, 3]
    height: int
        height of transformed image
    height: int
        width of transformed image

    Returns
    -------
    coords: ndarray
        perspective coordinates. shape is [height, width, 2]
    """
    coords = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # shape = [height, width, 2] after transpose
    coords = np.array(coords).transpose([1, 2, 0])
    ones = np.ones([height, width, 1])
    # homogeneous  coordinates
    coords = np.concatenate((coords, ones), axis=2)
    # transformed coordinates
    coords = coords @ transform_matrix
    # projection of coordinates
    coords[:, :, 0:2] /= coords[:, :, 2:]
    return coords[:, :, 0:2]

def yaw(t,imu_y,start_p):
    seta = math.atan2(f_x,start_p.x-m_x)
    new_seta = seta + imu_y * t
    new_u = m_x+f_x/math.tan(new_seta)
    if m_y-start_p.y == 0:
        det_v = 0
    else:
        tan_fai = f_y/math.sin(seta)/(m_y-start_p.y)
        new_v = m_y-f_y/tan_fai/math.sin(new_seta)
        det_v = new_v - start_p.y
    det_u = new_u - start_p.x
    return point2d(det_u,det_v)

def pitch(t,imu_x,start_p):
    seta = math.atan2(f_y,start_p.y-m_y)
    new_seta = seta - imu_x * t
    new_v = m_y+f_y/math.tan(new_seta)
    if start_p.x-m_x == 0:
        det_u = 0
    else:
        tan_fai = f_x/math.sin(seta)/(start_p.x-m_x)
        new_u = m_x+f_x/tan_fai/math.sin(new_seta)
        det_u = new_u - start_p.x
    det_v = new_v - start_p.y
    return point2d(det_u,det_v)

def roll(t,imu_z,start_p):
    r = math.sqrt((start_p.x - m_x)**2+(start_p.y - m_y)**2)
    seta = math.atan2(m_y - start_p.y,start_p.x - m_x)
    new_seta = seta + imu_z * t
    new_u = m_x + r * math.cos(new_seta)
    new_v = m_y - r * math.sin(new_seta)
    det_v = new_v - start_p.y
    det_u = new_u - start_p.x
    return point2d(det_u,det_v)

def distance_point(point_1,point_2):
    return math.sqrt((point_1.x-point_2.x)**2+(point_1.y-point_2.y)**2)

def Homography(result,id):
    points = np.array(data_h_point)
    uv = np.zeros([4,2],np.float32)
    uv[:,0] = points[:,1]
    uv[:,1] = points[:,0]
    uv = np.array(uv)
    res_points = []
    nodes = []
    for p in points:
        n = (p[1])/gap_w*(h/gap_h+1)+(p[0])/gap_h
        n = round(n)
        node = result[n]
        nodes.append(node)
        # res_points.append(get_track(point,result))
    n = 0
    ds = []
    distance = 0
    for ps in nodes:
        if distance_point(ps[id],ps[id+1])>distance:
            distance = distance_point(ps[id],ps[id+1])
    n = math.ceil(distance)
    if n == 0:
        n = 1
            
    Hs = []
    dst_points = []
    
    for i in range(0,n):
        for node in nodes:
            p = point2d(node[id+1].x*i/n+node[id].x*(n-i)/n,node[id+1].y*i/n+node[id].y*(n-i)/n)
            dst_points.append((p.x,p.y))
        dst_points = np.array(dst_points)
        dst_points = np.float32(dst_points[np.newaxis, :])
        H = dst_points
        Hs.append(H)
        dst_points = []
    return uv,Hs,n

# def Homography(result):
#     points = np.array(data_h_point)
#     uv = np.zeros([4,2],np.float32)
#     uv[:,0] = points[:,1]
#     uv[:,1] = points[:,0]
#     uv = np.array(uv)
#     res_points = []
#     for p in points:
#         point = point2d(p[1],p[0])
#         res_points.append(get_track(point,result))
#     n = 0
#     for ps in res_points:
#         if len(ps)>n:
#             n = len(ps)
#     Hs = []
#     dst_points = []
    
#     for i in range(1,n):
#         for res_point in res_points:
#             k = i/n*(len(res_point)-1)
#             k_1 = math.floor(k)
#             if k_1 != k:
#                 p = point2d(res_point[k_1].x+(k-k_1)*(res_point[k_1+1].x-res_point[k_1].x),res_point[k_1].y+(k-k_1)*(res_point[k_1+1].y-res_point[k_1].y))
#             else:
#                 p = res_point[k_1]
#             dst_points.append((p.x,p.y))
#         dst_points = np.array(dst_points)
#         dst_points = np.float32(dst_points[np.newaxis, :])
#         h = dst_points
#         Hs.append(h)
#         dst_points = []
#     return uv,Hs,n

def imu_line(imu):
    # print(imu)
    result = []
    for point in data_start_point:
        point = point2d(point[1],point[0])
        # point = point2d(339,31)
        ps = []
        ps.append(point2d(point.x,point.y))
        time = 0
        for one_imu in imu:
            if time == 0:
                time = one_imu[0]
                continue
            else:
                t = (one_imu[0]-time)/1000000
                d_1 = pitch(t,one_imu[1],point)
                d_2 = yaw(t,one_imu[2],point)
                d_3 = roll(t,one_imu[3],point)
                point.x = point.x + d_1.x + d_2.x + d_3.x
                point.y = point.y + d_1.y + d_2.y + d_3.y
                time = one_imu[0]
                ps.append(point2d(int(point.x),int(point.y)))
        result.append((ps))
    return result

def get_imu_data(data_gyro,time):
    imu = []
    time_img = time
    i = 0
    for time_imu in data_gyro[:, 0]:
        time_imu = time_imu / 1000
        if time_imu < int(time_img) - sep_time:
            i = i + 1
        else:
            time = data_gyro[i-1:i + 6, 0]/1000
            imu_x = data_gyro[i-1:i + 6, 1]
            imu_y = data_gyro[i-1:i + 6, 2]
            imu_z = data_gyro[i-1:i + 6, 3]

    for t, x, y, z in zip(time, imu_x, imu_y, imu_z):
        imu.append([t, x, y, z])

    # print(np.shape(testX))
    #result=get_blur_line(testX)
    result = imu_line(imu)
    # print(np.shape(result))
    return result

def new_blur(image,result):
    light = light_select(image)
    show_image = np.zeros([h,w,3],np.float64)
    img = np.zeros([h,w,3],np.float64)
    img_light = np.zeros([h,w,3],np.float64)
    line_len = 0
    for i in range(6):
        uv,Hs,n = Homography(result,i)
        line_len+=n
        #np.maximum(show_image,light)
        for H in Hs:
            H = np.array(H[0])
            transform_matrix = perspective_transform_matrix(uv, H)
            coords = perspective_coordinates(transform_matrix, h, w)
            img = nearest_sampler(image, coords)
            img_light = nearest_sampler(light, coords)
            # print(img.shape)
            np.clip(np.add(show_image,img/(6*n)),0,255,show_image)
            #np.maximum(show_image,img_light)
    return show_image,line_len

def light_select(image):
    showimage = np.zeros([h,w,3],np.float64)
    image = np.array(image)
    red = np.where(image[:,:,0]>v)
    blue = np.where(image[:,:,2]>v)
    green = np.where(image[:,:,1]>v)
    #print(red)
    blub = np.append(red,blue,axis=1)
    blub = np.append(blub,green,axis=1)
    for i in range(np.shape(blub)[1]):
        y = blub[0,i]
        x = blub[1,i]
        showimage[y,x] = image[y,x]
    return showimage


def het_map(image,start_point,value):
    for x in range(start_point.x,start_point.x+5):
        for y in range(start_point.y,start_point.y+5):
            image[y][x] = [127+value.x-start_point.x,127+value.y-start_point.y,0]

def blur(result):
    cp_img = np.zeros([h,w,3],np.float32)
    ep_img = np.zeros([h,w,3],np.float32)
    line_shape = 0
    for i in range(0,w,gap_w):
        for j in range(0,h,gap_h):
            start_point = point2d(i,j)
            line_s = line(0,0,0,0,[],[],points=get_track(start_point,result))
            c_point = fit(line_s)
            het_map(cp_img,start_point,c_point)
            e_point = line_s.points[-1]
            het_map(ep_img,start_point,e_point)
            # if int((i+j)/gap) % 2 == 0:
            #     left_point = point2d(i,j)
            #     right_point = point2d(i+gap,j+gap)
            #     line_shape = 1#左上右下对角点
            # elif int((i+j)/gap) % 2 == 1:
            #     left_point = point2d(i,j+gap)
            #     right_point = point2d(i+gap,j)
            #     line_shape = 2#左下右上对角点
            # line_1 = get_track(left_point,result)
            # line_2 = get_track(right_point,result)  
            # n_1 = len(line_1)-1
            # n_2 = len(line_2)-1
            # n = 7
            # # print(n)
            # block = np.zeros([gap,gap,3],np.uint8)
            # block = image[j:j+gap,i:i+gap]
            # for k in range(0,n+1):#图片区域进行模糊化
            #     p_1 = line_1[int(n_1*k/n)]
            #     p_2 = line_2[int(n_2*k/n)]
            #     ty = 127
            #     tx = 77
            #     block = cv2.resize(block,(abs(p_2.x-p_1.x),abs(p_2.y-p_1.y)))
            #     if p_1.x<0 and p_2.x<0 and p_1.y<0 and p_2.y < 0:
            #         continue
            #     #reshape工作
            #     # print(np.shape(block))
            #     if line_shape == 1:
            #         if np.shape(block)!=np.shape(show_image[p_1.y:p_2.y,p_1.x:p_2.x]):
            #             continue 
            #         np.clip(np.add(show_image[p_1.y:p_2.y,p_1.x:p_2.x],block/(n+1)),0,255,show_image[p_1.y:p_2.y,p_1.x:p_2.x])
            #         # if p_1.y <= ty and p_2.y > ty and p_1.x <= tx and p_2.x > tx:
            #         #     for y in range(ty-1,ty+2):
            #         #         for x in range(tx-1,tx+2):
            #         #             print(show_image[y,x])
            #         #     print(n)
            #         #     print(left_point.x,left_point.y)
            #         #     print(right_point.x,right_point.y)
            #         #     cv2.waitKey(100)
            #     elif line_shape == 2:
            #         if np.shape(block)!=np.shape(show_image[p_2.y:p_1.y,p_1.x:p_2.x]):
            #             continue 
            #         np.clip(np.add(show_image[p_2.y:p_1.y,p_1.x:p_2.x],block/(n+1)),0,255,show_image[p_2.y:p_1.y,p_1.x:p_2.x])
                    # if p_2.y <= ty and p_1.y > ty and p_1.x <= tx and p_2.x > tx:
                    #     for y in range(ty-1,ty+2):
                    #         for x in range(tx-1,tx+2):
                    #             print(show_image[y,x])
                        # print(n)
                        # print(left_point.x,left_point.y)
                        # print(right_point.x,right_point.y)
                        # cv2.waitKey(100)
                # cv2.imshow("image",show_image)
                # cv2.waitKey(1)
    # show_image = show_image.astype(np.uint8)
    return cp_img, ep_img
                

def read_image():
    res_list = os.listdir(res_img_path)
    i = 0
    c = 0
    while True:
        for res in res_list:
            # if i == len(res_list):
                # break
            # res = "roll"
            with open(path_test_data + res + path_imu, encoding='utf-8') as f:
                data_gyro = np.loadtxt(f, float, delimiter=",", skiprows=1, usecols=(0, 17, 18, 19))
            res_img_list = os.listdir(res_img_path+'/'+res)
            for res_img in res_img_list:
                time = res_img[0:16]
                # time = "1679401704231920"
                image = cv2.imread(img_path+img_list[i])
                # image = np.zeros([480,640,3],np.float64)
                # for j in range(20,w-20,50):
                #     for k in range(15,h-15,50):
                #         image[k,j]=[255,255,255]
                # shutil.copyfile(res_img_path+"/"+res+"/"+res_img,out_img_path+str(i+1).zfill(5)+".jpg")

                #os.rename(img_path+img_list[i-5601],img_path+str(i+1).zfill(4)+".jpg")
                result = get_imu_data(data_gyro,time)
                print(i)
                cp_image, ep_image = blur(result) 
                blur_image,n = new_blur(image,result)
                if n>0 :
                    c+=1
                    cv2.imwrite(out_img_path+str(i+1).zfill(4)+".jpg",blur_image)
                    cv2.imwrite(cp_img_path+str(i+1).zfill(4)+".jpg",cp_image)
                    cv2.imwrite(ep_img_path+str(i+1).zfill(4)+".jpg",ep_image)
                    cv2.imwrite(new_img_path+str(i+1).zfill(4)+".jpg",image)
                i+=1

def get_track(start_point,result):
    
    line_one = []
    point = start_point
    x = point.x
    y = point.y
    n = (x)/gap_w*(h/gap_h+1)+(y)/gap_h
    n = round(n)
    ps = result[n]
    line_one = compute_line(ps)
    line_one.append(ps[-1])
    # print("start")
    # for p in line:
    #     print(p.x,p.y)
    # line = []
    # for i in range(0,20):
    #     p = point2d(x+i,y+i)
    #     line.append(p)
    return line_one

def compute_line(points):
    # print("start")
    # for p in points:
        # print(p.x,p.y)
    old_point = point2d(-1000,-1000)
    # print("end")
    line_one = []
    for p in points:
        if old_point.x == -1000:
            old_point = p
            continue
        if p.x-old_point.x==0 and p.y-old_point.y==0:
            # print(p.x,p.y)
            continue      
        elif abs(p.x-old_point.x)>abs(p.y-old_point.y):
            a = (p.y-old_point.y) / (p.x-old_point.x)
            b = p.y - p.x * a
            if old_point.x>p.x:
                for i in range(old_point.x,p.x,-1):
                    line_one.append(point2d(i,round(a*i+b)))
                    # print(i,round(a*i+b))
            else:
                for i in range(old_point.x,p.x):
                    line_one.append(point2d(i,round(a*i+b)))
                    # print(i,round(a*i+b))
        else:
            a = (p.x-old_point.x) / (p.y-old_point.y)
            b = p.x - p.y * a
            if old_point.y>p.y:
                for i in range(old_point.y,p.y,-1):
                    line_one.append(point2d(round(a*i+b),i))
                    # print(round(a*i+b),i)
            else:
                for i in range(old_point.y,p.y):
                    line_one.append(point2d(round(a*i+b),i))
                    # print(round(a*i+b),i)
        old_point = p
    return line_one

if __name__ == "__main__":
    read_image()
    cut.cut()
    

