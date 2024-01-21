import math

num = 8
all_min_d = 0

class point2d(object):
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

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

def reinit_pointc(point,point1,point2,line):
    x = 0.0
    y = 0.0
    n = len(line.points)
    n = max(n//5,2)
    for p in line.points[0:n]:
        x += p.x
        y += p.y
    point1 = line.points[0]
    x_1 = x/n
    y_1 = y/n

    x = 0.0
    y = 0.0
    for p in line.points[-n-1:-1]:
        x += p.x
        y += p.y
    point2 = line.points[-1]
    x_2 = x/n
    y_2 = y/n

    a1= (y_1-point1.y)
    b1= (point1.x-x_1)
    c1= x_1 * point1.y - y_1 * point1.x
    a2= (y_2-point2.y)
    b2= (point2.x-x_2)
    c2= x_2 * point2.y - y_2 * point2.x
    det= a1*b2 - a2*b1
    if det==0:
        t = 1/2
        p_x = (point.x-point1.x*(1-t)**2-point2.x*t**2)/(2*t*(1-t))
        p_x = round(p_x)
        p_y = (point.y-point1.y*(1-t)**2-point2.y*t**2)/(2*t*(1-t))
        p_y = round(p_y)
        now_point = point2d(p_x,p_y)
    else:
        now_point = point2d(round((c2*b1 - c1*b2)/det),round((a2*c1 - a1*c2)/det))
    return now_point

def init_pointc(point,point1,point2,line):
    x = 0.0
    y = 0.0
    n = len(line.points)
    n = max(n//5,2)
    for p in line.points[0:n]:
        x += p.x
        y += p.y
    point1 = line.points[0]
    x_1 = x/n
    y_1 = y/n

    x = 0.0
    y = 0.0
    for p in line.points[-n-1:-1]:
        x += p.x
        y += p.y
    point2 = line.points[-1]
    x_2 = x/n
    y_2 = y/n

    m_x = point.x
    m_y = point.y
    dx_1 = point1.x - m_x
    dy_1 = point1.y - m_y
    dx_2 = point2.x - m_x
    dy_2 = point2.y - m_y



    a = angle(dx_1,dy_1,dx_2,dy_2)
    # print(a)
    if a < math.pi/6 or n<4:
        t = 1/2
        p_x = (point.x-point1.x*(1-t)**2-point2.x*t**2)/(2*t*(1-t))
        p_x = round(p_x)
        p_y = (point.y-point1.y*(1-t)**2-point2.y*t**2)/(2*t*(1-t))
        p_y = round(p_y)
        now_point = point2d(p_x,p_y)
    else :
        a1= (y_1-point1.y)
        b1= (point1.x-x_1)
        c1= x_1 * point1.y - y_1 * point1.x
        a2= (y_2-point2.y)
        b2= (point2.x-x_2)
        c2= x_2 * point2.y - y_2 * point2.x
        det= a1*b2 - a2*b1
        if det==0:
            t = 1/2
            p_x = (point.x-point1.x*(1-t)**2-point2.x*t**2)/(2*t*(1-t))
            p_x = round(p_x)
            p_y = (point.y-point1.y*(1-t)**2-point2.y*t**2)/(2*t*(1-t))
            p_y = round(p_y)
            now_point = point2d(p_x,p_y)
        else:
            now_point = point2d(round((c2*b1 - c1*b2)/det),round((a2*c1 - a1*c2)/det))
    return now_point

def distance(point1,point2):
    return (point1.x-point2.x)**2+(point1.y-point2.y)**2

def compute_bezier(point_c,point1,point2,only_line):
    k_c = 0
    sum_distance = 0
    max_distance = 0
    n = len(only_line.points)

    for i in range(1,n//2):
        t = i/(n//2)
        p_x = point1.x*(1-t)**2+point_c.x*t*(1-t)*2+point2.x*t**2
        p_y = point1.y*(1-t)**2+point_c.y*t*(1-t)*2+point2.y*t**2
        p = point2d(p_x,p_y)
        min_distance = 200
        for k in range(k_c,n):
            d = distance(only_line.points[k],p)
            if d < min_distance:
                min_distance = d
            else:
                k_c = k - 1
                break
        sum_distance += min_distance
        if min_distance > max_distance:
            max_distance = min_distance

    
    # if max_distance > 10:
    #     sum_distance = max_distance * (n-2)
    return sum_distance

def found_min_pointc(point_c,point1,point2,only_line):
    point_cr = []
    point_cr.append(point2d(point_c.x-1,point_c.y))
    point_cr.append(point2d(point_c.x-1,point_c.y-1))
    point_cr.append(point2d(point_c.x,point_c.y+1))
    point_cr.append(point2d(point_c.x-1,point_c.y+1))
    point_cr.append(point2d(point_c.x+1,point_c.y))
    point_cr.append(point2d(point_c.x+1,point_c.y-1))
    point_cr.append(point2d(point_c.x,point_c.y-1))
    point_cr.append(point2d(point_c.x+1,point_c.y+1))
    d = []
    for i in range(0,8):
        d.append(compute_bezier(point_cr[i],point1,point2,only_line))

    min_d = compute_bezier(point_c,point1,point2,only_line)
    min_i = 8
    for i in range(0,8):
        if d[i] < min_d:
            min_d = d[i]
            min_i = i
    if min_i < 8:
        return point_cr[min_i]
    else:
        return point_c
    

def fit(only_line):
    n = len(only_line.points)
    point1 = only_line.points[0]
    point2 = only_line.points[-1]
    n = len(only_line.points)
    point = only_line.points[n//2]
    point_c = init_pointc(point,point1,point2,only_line)
    while 1:
        new_point_c = found_min_pointc(point_c,point1,point2,only_line)
        if new_point_c == point_c:
            break
        else:
            point_c = new_point_c
    return point_c

def check_fit(only_line):
    c = 0
    n = len(only_line.points)
    point1 = only_line.points[0]
    point2 = only_line.points[-1]
    n = len(only_line.points)
    point = only_line.points[n//2]
    point_c = init_pointc(point,point1,point2,only_line)
    points = []
    points.append(point_c)
    while 1:
        new_point_c = found_min_pointc(point_c,point1,point2,only_line)
        if new_point_c == point_c:
            d = compute_bezier(point_c,point1,point2,only_line)
            if d > n:
                print(point_c.x,point_c.y,d)
                reinit_pointc(point,point1,point2,only_line)
                c += 1
                if c>3:
                    break
                continue
            break
        else:
            point_c = new_point_c
            points.append(point_c)
        
    return points


def draw_point(only_line):
    ps = []
    for i in range(0,6):
        t = i/5
        point1 = only_line.points[0]
        point2 = only_line.points[-1]
        point_c = only_line.ctrls[-1]
        p_x = point1.x*(1-t)**2+point_c.x*t*(1-t)*2+point2.x*t**2
        p_y = point1.y*(1-t)**2+point_c.y*t*(1-t)*2+point2.y*t**2
        p = point2d(round(p_x),round(p_y))
        ps.append(p)
    return ps
