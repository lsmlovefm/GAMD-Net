import os
import numpy as np 
import math
import cv2
import pandas as pd
import shutil
from bezier import fit,check_fit
from bezier import draw_point
from bezier import point2d



def gougu(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def angle(dx_1,dy_1,dx_2,dy_2):
    if dx_1 == 0 and dx_2 == 0:
        return 0
    elif dx_1 == 0:
        a = math.atan(dy_2/dx_2)
        return math.pi/2 - abs(a)
    elif dx_2 == 0:
        a = math.atan(dy_1/dx_1)
        return math.pi/2 - abs(a)
    elif dx_1*dx_2 + dy_1*dy_2 == 0:
        a = math.pi/2
    else:
        a = math.atan((dy_1/dx_1-dy_2/dx_2)/(1+dy_1*dy_2/(dx_1*dx_2)))
    return abs(a) 

def point(line_1,line_2):
    a1= 2*(line_1.two_point_x-line_1.one_point_x)
    b1= 2*(line_1.two_point_y-line_1.one_point_y)
    c1= line_1.one_point_x**2 - line_1.two_point_x**2 + line_1.one_point_y**2 - line_1.two_point_y**2
    a2= 2*(line_2.two_point_x-line_2.one_point_x)
    b2= 2*(line_2.two_point_y-line_2.one_point_y)
    c2= line_2.one_point_x**2 - line_2.two_point_x**2 + line_2.one_point_y**2 - line_2.two_point_y**2
    det= a1*b2 - a2*b1
    p = []
    p.append((c2*b1 - c1*b2)/det)
    p.append((a2*c1 - a1*c2)/det)
    # print(p)
    return p

class line(object):
    def __init__(self,one_point_x,one_point_y,two_point_x,two_point_y,u_xs,u_ys,points) -> None:
        self.one_point_x = one_point_x
        self.one_point_y = one_point_y
        self.two_point_x = two_point_x
        self.two_point_y = two_point_y
        self.points = points
        self.u_xs = u_xs
        self.u_ys = u_ys

        self.ctrl = point2d(0,0)
        self.ctrls = []




def start_point(path):
    csv_path = path+"res_important_line/"
    imu_file = "gyro.csv"
    img_path = path+"res_important_show/"#线条图片文件路径
    start_point_path = path+"start_point/"
    end_point_path = path+"end_point/"

    folder = os.path.exists(start_point_path)
    if folder:
        shutil.rmtree(start_point_path)
    os.makedirs(start_point_path)

    folder = os.path.exists(end_point_path)
    if folder:
        shutil.rmtree(end_point_path)
    os.makedirs(end_point_path)

    with open(path+imu_file,encoding = 'utf-8') as f:
        data_gyro = np.loadtxt(f,float,delimiter = ",", skiprows = 1,usecols = (0,17,18,19))

    #csvlist = os.listdir(path)
    csvlist = os.listdir(img_path)
    imglist = os.listdir(img_path)
    sep_time = 80000
    for csv in csvlist: #读取存储轨迹坐标的csv文件
        i = 0
        csv = csv[0:16]+'.csv'
        start_points_x = []
        start_points_y = []
        end_points_x = []
        end_points_y = []
        if csv == "gyro.csv":
            continue
        time_img = csv[0:16]
        time_img = int(time_img)
        time = 0
        print(time_img)
        
        # print(data_gyro[:,0])
        for time_imu in data_gyro[:,0]:
            time_imu = time_imu/1000
            if time_imu < time_img - sep_time:
                i = i+1
            else:
                imu_x = data_gyro[i,1]
                imu_y = data_gyro[i,2]
                imu_z = data_gyro[i,3]
                # print(time_imu)
                # print(time_img - sep_time)
                # print(i)
                # print(imu_x,imu_y,imu_z)
                break
            time = time_imu 
            
        with open(csv_path+csv,encoding = 'utf-8') as f:
            data_lines = np.loadtxt(f,int,delimiter = ",", skiprows = 0,usecols = (0,1))
        data_x = data_lines[:,1]
        data_y = data_lines[:,0]
        lines = []
        u_ys = []
        u_xs = []
        points = []
        u_x = 0 # 1上，-1下
        u_y = 0
        old_x = 0
        old_y = 0
        one_point_x = 0
        one_point_y = 0
        for x,y in zip(data_x,data_y):
            if old_x == 1 and old_y == 0:
                one_point_x = x
                one_point_y = y

            elif x == 0 and y == 0:#当前点为0，0
                if old_x!=0 and old_y!=0:
                    two_point_x = old_x
                    two_point_y = old_y
                    # print(len(points))
                    # print(u_ys)
                    u_xs = np.array(u_xs)
                    u_ys = np.array(u_ys)
                    if one_point_y > two_point_y:
                        u_xs = u_xs[::-1]
                        u_xs = -u_xs
                    if one_point_x > two_point_x:
                        u_ys = u_ys[::-1]
                        u_ys = -u_ys
                    lines.append(line(one_point_x,one_point_y,two_point_x,two_point_y,u_xs,u_ys,points))
                    points = []
                    u_ys = []
                    u_xs = []
                    u_x = 0 # 1上，-1下
                    u_y = 0
                    
                    
                    
            elif old_x != 0 and old_y != 0:#当前点不是起点或为（0，0）
                if y != 0:
                    p = point2d(x,y)
                    points.append(p)
                if x - old_x > 0 and u_x != 1:
                    u_x = 1
                    u_xs.append(u_x)
                elif x - old_x < 0 and u_x != -1:
                    u_x = -1
                    u_xs.append(u_x)
                elif y - old_y > 0 and u_y != 1:
                    u_y = 1
                    u_ys.append(u_y)
                elif y - old_y < 0 and u_y != -1:
                    u_y = -1
                    u_ys.append(u_y)
            old_x = x
            old_y = y
        # for l in lines:
        #     print(l.u_ys)

        # is_inclined_l = 0
        # is_inclined_r = 0s
        for l in lines:
            #p = fit(l) #检查控制点移动轨迹
            ps = check_fit(l)
            # x = p.x
            # y = p.y
            #l.ctrl = p
            l.ctrls = ps

        is_vertical = 0
        is_horizontal = 0
        is_roll = 0
        for l in lines:
            det_x = l.one_point_x - l.two_point_x
            det_y = l.one_point_y - l.two_point_y
            # if not (( det_x > 0 and det_y > 0) or ( det_x < 0 and det_y < 0 )):
            #     is_inclined_l = 1
            # if not (( det_x < 0 and det_y > 0) or ( det_x > 0 and det_y < 0 )):
            #     is_inclined_r = 1
            if not abs(det_x) >= abs(det_y):
                is_horizontal = 1
            if not abs(det_x) <= abs(det_y):
                is_vertical = 1
        # print(is_inclined_l,is_inclined_r,is_horizontal,is_vertical)

                
        if is_vertical == 0:
            if imu_x > 0:
                #上面的点是起点
                for l in lines:
                    if l.one_point_y > l.two_point_y:
                        point_x = l.two_point_x
                        point_y = l.two_point_y
                        l.two_point_x = l.one_point_x
                        l.two_point_y = l.one_point_y
                        l.one_point_x = point_x
                        l.one_point_y = point_y
            
            else:
                for l in lines:
                    if l.one_point_y < l.two_point_y:
                        point_x = l.two_point_x
                        point_y = l.two_point_y
                        l.two_point_x = l.one_point_x
                        l.two_point_y = l.one_point_y
                        l.one_point_x = point_x
                        l.one_point_y = point_y

        elif is_horizontal == 0:
            if imu_y > 0:
                #右边的点是起点
                for l in lines:
                    if l.one_point_x < l.two_point_x:
                        point_x = l.two_point_x
                        point_y = l.two_point_y
                        l.two_point_x = l.one_point_x
                        l.two_point_y = l.one_point_y
                        l.one_point_x = point_x
                        l.one_point_y = point_y
            else:
                for l in lines:
                    if l.one_point_x > l.two_point_x:
                        point_x = l.two_point_x
                        point_y = l.two_point_y
                        l.two_point_x = l.one_point_x
                        l.two_point_y = l.one_point_y
                        l.one_point_x = point_x
                        l.one_point_y = point_y
        else:
            old_ux = []
            old_uy = []
            symbol_x = 0
            symbol_y = 0
            for l in lines:
                # print(l.u_xs)
                # print(l.u_ys)
                l.u_xs = list(l.u_xs)
                l.u_ys = list(l.u_ys)
                if len(old_ux) == 0 and len(old_uy) == 0:
                    old_ux = l.u_xs
                    old_uy = l.u_ys
                    continue
                else:
                    # print(l.u_xs,old_ux)
                    # print(l.u_ys,old_uy)
                    if l.u_xs != old_ux and symbol_x == 0:
                        symbol_x = 1
                    if l.u_ys != old_uy and symbol_y == 0:
                        symbol_y = 1
                    old_ux = l.u_xs
                    old_uy = l.u_ys
            # print(symbol_x,symbol_y)
            if symbol_x == 0:
                if imu_x > 0:
                    #上面的点是起点
                    for l in lines:
                        if l.one_point_y > l.two_point_y:
                            point_x = l.two_point_x
                            point_y = l.two_point_y
                            l.two_point_x = l.one_point_x
                            l.two_point_y = l.one_point_y
                            l.one_point_x = point_x
                            l.one_point_y = point_y
                
                else:
                    for l in lines:
                        if l.one_point_y < l.two_point_y:
                            point_x = l.two_point_x
                            point_y = l.two_point_y
                            l.two_point_x = l.one_point_x
                            l.two_point_y = l.one_point_y
                            l.one_point_x = point_x
                            l.one_point_y = point_y

            elif symbol_y == 0:
                if imu_y > 0:
                    #右边的点是起点
                    for l in lines:
                        if l.one_point_x < l.two_point_x:
                            point_x = l.two_point_x
                            point_y = l.two_point_y
                            l.two_point_x = l.one_point_x
                            l.two_point_y = l.one_point_y
                            l.one_point_x = point_x
                            l.one_point_y = point_y
                else:
                    for l in lines:
                        if l.one_point_x > l.two_point_x:
                            point_x = l.two_point_x
                            point_y = l.two_point_y
                            l.two_point_x = l.one_point_x
                            l.two_point_y = l.one_point_y
                            l.one_point_x = point_x
                            l.one_point_y = point_y

            else:
                is_roll = 1
                max_a = np.zeros(4)
                lines_1 = np.zeros(4,dtype = line)
                lines_2 = np.zeros(4,dtype = line)
                for i in range(len(lines)):
                    for j in range(i+1,len(lines)):
                        l_1 = lines[i]
                        l_2 = lines[j]
                        l_1dx = l_1.one_point_x - l_1.two_point_x
                        l_1dy = l_1.one_point_y - l_1.two_point_y
                        l_2dx = l_2.one_point_x - l_2.two_point_x
                        l_2dy = l_2.one_point_y - l_2.two_point_y
                        if abs(l_1dx) <= 6 and abs(l_1dy) <= 6:
                            break
                        if abs(l_2dx) <= 6 and abs(l_2dy) <= 6:
                            continue
                        line_a = angle(l_1dx,l_1dy,l_2dx,l_2dy)
                        for k in range(4):
                            if line_a > max_a[k]:
                                if k > 0:
                                    max_a[k-1] = max_a[k]
                                    lines_1[k-1] = lines_1[k]
                                    lines_2[k-1] = lines_2[k]           
                                max_a[k] = line_a
                                lines_1[k] = l_1
                                lines_2[k] = l_2
                # print(max_a)
                xs = []
                ys = []
                # if lines_1[0]==0 or lines_1[3]==0:
                #     os.remove(path+'res_important/'+str(time_img)+'.jpg')
                #     print("remove")
                #     continue
                for i in range(4):
                    point_2 = point(lines_1[i],lines_2[i])
                    xs.append(round(point_2[0]))
                    ys.append(round(point_2[1]))
                min = 100
                for i in range(0,4):
                    for j in range(i+1,4):
                        if(gougu(xs[i],ys[i],xs[j],ys[j]))<min:
                            min = gougu(xs[i],ys[i],xs[j],ys[j])
                            x_1 = xs[i]
                            x_2 = xs[j]
                            y_1 = ys[i]
                            y_2 = ys[j]
                y = round((y_1+y_2)/2)
                x = round((x_1+x_2)/2)
                # print(x,y)
                
                if imu_z < 0:
                    #顺时针旋转
                    for l in lines:
                        if abs(l.one_point_y-l.two_point_y) >= abs(l.one_point_x-l.two_point_x):
                            if l.one_point_x > x and l.two_point_x > x:
                                #y小的是起点
                                if l.one_point_y > l.two_point_y:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            elif l.one_point_x < x and l.two_point_x < x:
                                #y大的是起点
                                if l.one_point_y < l.two_point_y:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            else:
                                print("error")
                        else:
                            if l.one_point_y > y and l.two_point_y > y:
                                #x大的是起点
                                if l.one_point_x < l.two_point_x:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            elif l.one_point_y < y and l.two_point_y < y:
                                if l.one_point_x > l.two_point_x:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            else:
                                print("error")
                else:
                    #逆时针旋转
                    for l in lines:
                        if abs(l.one_point_y-l.two_point_y) >= abs(l.one_point_x-l.two_point_x):
                            if l.one_point_x > x and l.two_point_x > x:
                                #y大的是起点
                                if l.one_point_y < l.two_point_y:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            elif l.one_point_x < x and l.two_point_x < x:
                                #y小的是起点
                                if l.one_point_y > l.two_point_y:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            else:
                                print('error')
                        else:
                            if l.one_point_y > y and l.two_point_y > y:
                                #x小的是起点
                                if l.one_point_x > l.two_point_x:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            elif l.one_point_y < y and l.two_point_y < y:
                                if l.one_point_x < l.two_point_x:
                                    point_x = l.two_point_x
                                    point_y = l.two_point_y
                                    l.two_point_x = l.one_point_x
                                    l.two_point_y = l.one_point_y
                                    l.one_point_x = point_x
                                    l.one_point_y = point_y
                            else:
                                print('error')

        for i in range(len(lines)):
            for j in range(i+1,len(lines)):
                if abs(lines[i].one_point_x - lines[j].one_point_x) == 2 and abs(lines[i].one_point_y - lines[j].one_point_y) < 2:
                    print("warning")
                if abs(lines[i].one_point_y - lines[j].one_point_y) == 2 and abs(lines[i].one_point_x - lines[j].one_point_x) < 2:
                    print("warning")
                if abs(lines[i].one_point_x - lines[j].two_point_x) == 2 and abs(lines[i].one_point_y - lines[j].two_point_y) < 2:
                    print("warning")
                if abs(lines[i].one_point_y - lines[j].two_point_y) == 2 and abs(lines[i].one_point_x - lines[j].two_point_x) < 2:
                    print("warning")
                if abs(lines[i].two_point_x - lines[j].one_point_x) == 2 and abs(lines[i].two_point_y - lines[j].one_point_y) < 2:
                    print("warning")
                if abs(lines[i].two_point_y - lines[j].one_point_y) == 2 and abs(lines[i].two_point_x - lines[j].one_point_x) < 2:
                    print("warning")
                if abs(lines[i].two_point_x - lines[j].two_point_x) == 2 and abs(lines[i].two_point_y - lines[j].two_point_y) < 2:
                    print("warning")
                if abs(lines[i].two_point_y - lines[j].two_point_y) == 2 and abs(lines[i].two_point_x - lines[j].two_point_x) < 2:
                    print("warning")

        if is_roll == 1:
            # print("roll")
            for img in imglist:
                if float(img[0:17]) == time_img:
                    image = cv2.imread(img_path+img)
                    cv2.circle(image,(x,y),1,(255,255,255),4)
                    
                    #     cv2.circle(image,(line_1.one_point_x,line_1.one_point_y),1,(0,0,255),4)
                    #     cv2.circle(image,(line_1.two_point_x,line_1.two_point_y),1,(0,0,255),4)
                    #     cv2.circle(image,(line_2.one_point_x,line_2.one_point_y),1,(0,0,255),4)
                    #     cv2.circle(image,(line_2.two_point_x,line_2.two_point_y),1,(0,0,255),4)
                    # cv2.imwrite(img_path+img,image)
                # cv2.imshow("image",image)
                # cv2.waitKey(0)
                
        for img in imglist:
            if float(img[0:17]) == time_img:
                image = cv2.imread(img_path+str(time_img)+'.jpg')
                start_points_y = []
                start_points_x = []
                end_points_x = []
                end_points_y = []
                ctrl_points_x = []
                ctrl_points_y = []
                for l in lines:
                    for p in draw_point(l):
                        if p.x<640 and p.y <480:
                        # cv2.circle(image,(p.x,p.y),1,(0,0,255),1)
                            image[p.y,p.x]=(0,255,0)
                    for p in l.ctrls:
                        if p.x<640 and p.y <480:
                            image[p.y,p.x]=(0,0,255)
                    p = l.ctrls[-1]
                    if p.x<640 and p.y <480:
                        image[p.y,p.x]=(255,0,0)   
                    start_points_x.append(l.one_point_x)
                    start_points_y.append(l.one_point_y)
                    end_points_x.append(l.two_point_x)
                    end_points_y.append(l.two_point_y)
                    ctrl_points_x.append(p.x)
                    ctrl_points_y.append(p.y)
                                        
                for x,y in zip(start_points_x,start_points_y):
                    cv2.circle(image,(x,y),1,(0,0,255),4)
                cv2.imwrite(img_path+img,image)
                df = pd.DataFrame(zip(start_points_y,start_points_x))
                df.to_csv(start_point_path+img[:16]+".csv",mode='a',header=True, index=None) 
                df = pd.DataFrame(zip(end_points_y,end_points_x,ctrl_points_y,ctrl_points_x))
                df.to_csv(end_point_path+img[:16]+".csv",mode='a',header=True, index=None) 
                # cv2.imshow("image",image)
                # cv2.waitKey(0)

        
if __name__ == "__main__":
    path = "data/"#csv轨迹文件路径
    path_list = os.listdir(path)
    for data_path in path_list:
        start_point(path+data_path+"/")    
    
                    
