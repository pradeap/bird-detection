import cv2
from ultralytics import YOLO
import os
import numpy as np
import math
global identified


model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture('media1.mp4')
fps=cam.get(cv2.CAP_PROP_FPS)

#Dirctory to save the image
output_folder = "/Users/yogeshthangamuthu/Desktop/project/Cv Library projects/BIRD PROJECT/bird_image"
os.makedirs(output_folder, exist_ok=True)

def filter(image):

    filtered_image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return filtered_image
def re_combine_close_rectangles(contours, threshold_distance=60):#closest objects from camera
    combined_rectangles = []

    for i, contour in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(contour)

        # Check against already combined rectangles
        merged = False
        for j, (x2, y2, w2, h2) in enumerate(combined_rectangles):
            # Calculate the distance between the centers of bounding rectangles
            distance = np.sqrt((x1 + w1/2 - x2 - w2/2)**2 + (y1 + h1/2 - y2 - h2/2)**2)

            if distance < threshold_distance:
                # Merge the rectangles
                combined_rectangles[j] = (
                    min(x1, x2), min(y1, y2),
                    max(x1 + w1, x2 + w2) - min(x1, x2),
                    max(y1 + h1, y2 + h2) - min(y1, y2)
                )
                merged = True
                break

        if not merged:
            # If not close to any existing rectangle, add it as a new rectangle
            combined_rectangles.append((x1, y1, w1, h1))

    combined_rectangles = [(x, y, w, h,0) for x, y, w, h in combined_rectangles if w >= 10 and h >= 10 and w*h >= 500]



    return combined_rectangles






def combine_close_rectangles(contours, threshold_scale=0.9):
    combined_rectangles = []
    temp=[]
    for i, contour in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        if y1+int(h1/2) >750:#boundary decider for closest object

            temp.append(contour)

        else:
            area1 = w1 * h1

            # Check against already combined rectangles
            merged = False

            for j, (x2, y2, w2, h2,a) in enumerate(combined_rectangles):
                a=2
                area2 = w2 * h2
                threshold_distance1 = threshold_scale * max(w1, h1, w2, h2)
                threshold_distance2 = threshold_scale * np.sqrt((area1 + area2) / 2)
                distance = np.sqrt((x1 + w1/2 - x2 - w2/2)**2 + (y1 + h1/2 - y2 - h2/2)**2)
                threshold_distance=max(threshold_distance1,threshold_distance2)
                if distance < threshold_distance:
                    w=max(x1 + w1, x2 + w2) - min(x1, x2)
                    h=max(y1 + h1, y2 + h2) - min(y1, y2)
                    if w*h >=1000:
                        a=1
                    combined_rectangles[j] = (
                        min(x1, x2), min(y1, y2),
                        w,h,a
                    )

                    merged = True
                    break

            if not merged:
                # If not close to any existing rectangle, add it as a new rectangle
                combined_rectangles.append((x1, y1, w1, h1,2))
    temp=re_combine_close_rectangles(temp)
    combined_rectangles= combined_rectangles + temp
    #combined_rectangles1=calculate_iou(combined_rectangles) #Avoid overlap tech
    return combined_rectangles
def calculate_iou1(box1, box2):
    x11, y11, x21, y21,a1 = box1
    x12, y12, x22, y22,a2 = box2
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])
    box1_area=(x21-x11)*(y21-y11)
    box2_area=(x22-x12)*(y22-y12)

    merge_area=(x2_int-x1_int)*(y2_int-y1_int)
    if ((x12<=x11<=x22) and (y12<=y11<=y22)) or ((x12<=x21<=x22) and (y12<=y21<=y22)) or ((x11<=x12<=x21) and (y11<=y12<=y21)) or ((x11<=x22<=x21) and (y11<=y22<=y21)):
        if box1_area >= box2_area:
            return ((box1_area - merge_area) / box1_area, 1)
        else:
            return ((box2_area - merge_area) / box2_area, 2)
    else:
        return 0,0



def filter_overlapping_rectangles(rectangles, overlap_threshold=0.2):
    filtered_rectangles = []

    for rect1 in rectangles:
        if len(filtered_rectangles)==0:
            filtered_rectangles.append(rect1)
        else:
            temp=[]
            t=0
            p=0
            for rect2 in filtered_rectangles:

                iou,pos=calculate_iou1(rect1,rect2)

                if pos==0:

                    temp.append(rect2)
                    t=1

                if pos==1 and iou>0:
                    p=1
                    temp.append(rect1)
                if pos==2 and iou>0:
                    p=1
                    temp.append(rect2)

            if t==1 and p==0:
                temp.append(rect1)

            filtered_rectangles=temp
    return filtered_rectangles
def calculate_distance(rect1, rect2):
    # Assuming rectangles are represented as (x, y, x1, y1, area, id)
    x1, y1, _, _, _= rect1
    x2, y2, _, _, _= rect2

    # Calculate Euclidean distance between rectangle centers
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance
def group_rectangles(rectangles, threshold):
    groups = []
    assigned_indices = set()

    for i, rect1 in enumerate(rectangles):
        if i not in assigned_indices:
            group = [i]
            for j, rect2 in enumerate(rectangles):
                if i != j and j not in assigned_indices:
                    distance = calculate_distance(rect1, rect2)
                    if distance < threshold:
                        group.append(j)
                        assigned_indices.add(j)

            groups.append(group)

    return groups
def side_overlap(contours):
    contours1=[]
    threshold=10
    groups = group_rectangles(contours, threshold)
    for i,val in enumerate(groups):
        x=100000
        x1=0
        y=100000
        y1=0
        a=0
        if len(val)>=2:
            for j in val:

                x=min(x,contours[j][0])
                y = min(y, contours[j][1])
                x1 = max(x1, contours[j][2])
                y1 = max(y1, contours[j][3])
                a = max(a, contours[j][4])
            contours1.append((x,y,x1,y1,a))



        else:
            val=val[0]
            contours1.append(contours[val])
    return contours1




def avoid_overlap(contours):
    contours1=[]
    for x,y,w,h,a in contours:
        x1=x+w
        y1=y+h
        contours1.append((x,y,x1,y1,a))
    filtered_rectangles = filter_overlapping_rectangles(contours1, overlap_threshold=0.4)
    filtered_rectangles = side_overlap(filtered_rectangles)
    filtered_rectangles1=[]

    for x,y,x1,y1,a in filtered_rectangles:
        w=x1-x
        h=y1-y
        filtered_rectangles1.append((x,y,w,h,a))

    return filtered_rectangles1
#def display(contours):

def overlap(prev,frame):

    kernel = np.array((9, 9), dtype=np.uint8)
    frame1 = prev
    frame2 = frame
    img1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    g_frame1=img1
    g_frame2=img2
    grayscale_diff = cv2.subtract(g_frame1, g_frame2)
    frame_diff = cv2.medianBlur(grayscale_diff, 3)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)


    frame_with_contours = frame2.copy()

    contours=combine_close_rectangles(contours)
    contours=avoid_overlap(contours)
    return contours
def direction_calculator(curr,prev):
    prev=list(prev[0])
    x1,y1=curr[0],curr[1]
    x2, y2 = prev[0], prev[1]
    x=abs(curr[0]-prev[0])
    y=abs(curr[1]-prev[1])
    if x>y:
        if x1>x2:
            return 2
        else:
            return 4
    else:
        if y1>y2:
            return 3
        else:
            return 1


def direction_condition(input,check):
    x1,y1=check[0],check[1]
    x2,y2=input[0],input[1]
    dr=check[12]
    if dr==4:
        if x2<x1 :
            return 1
    elif dr ==2:
        if x2>x1 :
            return 1
    elif dr ==3:
        if y2>y1:
            return 1
    else:
        if y1>y2:
            return 1
    return 0
def size_condition(input,check):#9
    a1=check[9]
    a2=input[9]
    k=.40
    if (a1 - (a1 * k)) <= a2 <= (a1 + (a1 * k)) or (a2 - (a2 * k)) <= a1 <= (a2 + (a2 * k)):
        return 1
    return 0

def id_checker(inp,check):
    w1=inp[4]
    h1=inp[5]
    templst=[]
    for i,lst in enumerate(check):
        x2,y2=lst[6],lst[7]
        x1,y1=inp[6],inp[7]
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        com=max(w1,h1)
        if distance<=com*2.5:
            templst.append(lst)

    if len(templst) == 1:
        identified.append(templst[0][10])
        k=templst[0][11]

        dir = direction_calculator(inp, templst)
        k += 1
        return (templst[0][10],k,dir)
    elif len(templst)>2:
        layer1=[]
        new=inp
        for i in templst:
            k=direction_condition(inp,i)
            l=size_condition(inp,i)
            if k==1 and l==1:

                return (i[10],i[11],i[12])
        return (0,0,0)

    else:
        if len(templst)==2:
            x,y=inp[0],inp[1]
            x1,y1=templst[0][0],templst[0][0]
            x2, y2 = templst[1][0], templst[1][0]
            id1=templst[0][10]#id for first
            id2 = templst[1][10]  # id for second
            fd1=templst[0][12]#direction of first
            fd2=templst[1][12]#direction of second
            if id1 not in identified and id2 not in identified:
                if fd1==fd2:
                    if fd1==4:
                        if x<x1:
                            val = templst[0]
                        else:
                            val=templst[1]
                    elif fd1==2:
                        if x>x1:
                            val = templst[0]
                        else:
                            val=templst[1]
                    elif fd1==1:
                        if y<y1:
                            val = templst[0]
                        else:
                            val=templst[1]
                    else:
                        if y>y1:
                            val = templst[0]
                        else:
                            val=templst[1]

                    new=[]
                    new.append(val)
                    dir = direction_calculator(inp, new)
                    k = new[0][11]
                    identified.append(new[0][10])
                    return (new[0][10], k, dir)




            else :
                if id1 not in identified:
                    id = id1
                    val=templst[0]
                elif id2 not in identified:
                    id = id2
                    val = templst[1]
                else:
                    return (0,0,0)
                new = []
                new.append(val)

                dir = direction_calculator(inp, new)
                k = new[0][11]
                identified.append(new[0][10])
                return (new[0][10], k, dir)
        return (0,0,0)

def find_it(id,val):
    temp=[]
    for i in id :
        for j in val:
            if j[10]==i:
                temp.append(j)
    return temp
def calculate_distance1(center1, center2):
    return int(math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2))

def align_input_with_memory(input_values, memory_lists):
    xc,yc=input_values[6],input_values[7]
    temp=[]
    temp1 = []
    dict_temp={}
    for i in memory_lists:
        temp.append(calculate_distance1((xc,yc),(i[6],i[7])))
    for i,j in enumerate(temp):
        dict_temp[j]=i
    temp.sort()
    for i in temp:
        temp1.append(memory_lists[dict_temp[i]])

    return temp1

def parity_check(input,check):#change
    temp=[]
    f1=input[13]
    f2=check[13]


    x1c=input[6]
    y1c = input[7]
    x2c=check[6]
    y2c = check[7]

    dist = ((x1c - x2c) ** 2 + (y1c - y2c) ** 2) ** 0.5
    xoff=max(input[4],input[5],check[4],check[5])
    ff = f1 - f2
    if f1-f2<=2:
        k=.30
        xoff=2*xoff
    else:
        k=.30+(.10*(ff-2))

        xoff=(2*xoff)*(ff)

    a1=input[9]
    a2=check[9]
    if dist <=xoff:
        if check[11]==0:

            A1=a1+(a1*k)
            A2=a2+(a2*k)

            if (a1 - (a1 * k)) <= a2 <= (a1 + (a1 * k)) or (a2 - (a2 * k)) <= a1 <= (a2 + (a2 * k)):

                d=direction_calculator(input,[check])

                checktemp=list(check)
                checktemp[12]=d
                e=direction_condition(input,checktemp)
                return e,d



        else:
            d = check[11]
            e = direction_condition(input, check)
            return e,d
    return 0, 0


def memory_checker(input,memory):

    temp=[]
    for i in memory:
        if len(i) !=0:

            i=align_input_with_memory(input,i)

            for j in i :

                e,d=parity_check(input,j)
                if e==1:
                    eid=j[10]
                    return d,eid
    return 0,0

def memory_remover(mem,id):
    mem_temp=[]
    for i in mem:
        if len(i)!=0:
            temp=[]
            for j in i:
                if j[10]!= id:
                    temp.append(j)
            mem_temp.append(temp)
        else :
            mem_temp.append(i)
    return mem_temp


def initial_count(lst):
     if lst[9]<=700:
         ref_line_pts = np.array([[0,750], [1000, 630]])

         # Calculate slope of the reference line
         slope = (ref_line_pts[1, 1] - ref_line_pts[0, 1]) / (ref_line_pts[1, 0] - ref_line_pts[0, 0])
         x, y = lst[6],lst[7]
         reference_y = int(slope * (x - ref_line_pts[0, 0]) + ref_line_pts[0, 1])
         print("slope",slope)
         print("reference_y",y < reference_y)

frame_index = 0

identified = []
id=1
prevframe=0
size_saver=[]
id_prev=[]
val_prev=[]
memory=[]
while True:
    temp=[]
    newpd=[]
    newval=[]
    print("frame_index",frame_index)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    ret, frame = cam.read()
    copy_frame=frame.copy()

    if not ret:
        break  # Break the loop if no more frames are available


    if frame_index == 0:
        prevframe=frame
        frame_index += 1
        continue  # Skip the first frame

    detection=overlap(prevframe,frame)


    for x,y,w,h,a in detection:
        xc = x + int(w / 2)
        yc = y + int(h / 2)
        x1=x+w
        y1=y+h
        are=w*h

        if are>=70:
            g = initial_count((x, y, x1, y1, w, h, xc, yc, a, are, 0, 0, 0, frame_index))
            if frame_index==1:

                b=id
                id+=1
                c=0
                g = initial_count((x,y,x1,y1,w,h,xc,yc,a,are,b,c,0,frame_index))
                temp.append((x,y,x1,y1,w,h,xc,yc,a,are,b,c,0,frame_index,g))
                newpd.append(b)
                newval.append((x,y,x1,y1,w,h,xc,yc,a,are,b,c,0,frame_index,g))
            else:
                view=id_checker((x,y,x1,y1,w,h,xc,yc,a,are,0,0,0,frame_index,g),size_saver)
                id_fet=view[0]
                c=view[1]
                d=view[2]
                if id_fet!=0:
                    b = id_fet
                    k=(x, y, x1, y1, w, h, xc, yc, a, are, b,c,d,frame_index,g)
                    temp.append(k)

                else:

                    d,b,=memory_checker((x,y,x1,y1,w,h,xc,yc,a,are,0,0,0,frame_index,g),memory)

                    if b==0 or d==0:
                        b=id
                        print("new_id",b)
                        id+=1
                    else:
                        memory=memory_remover(memory,b)

                    k=(x, y, x1, y1, w, h, xc, yc,a, are, b, 0, 0,frame_index,g)
                    temp.append(k)

                newpd.append(b)
                newval.append(k)
            rectangle_points=[(0,750), (50, 800), (1000,630), (950, 580)]
            cv2.line(copy_frame, (0,750), (1000,630), (0,0,255), 2)
            cv2.line(copy_frame, (1000, 630), (1920, 750), (0, 0, 255), 2)
            cv2.rectangle(copy_frame, (700,0), (1230,630), (0, 255, 0), 2)
            #cv2.polylines(frame, [np.array(rectangle_points)], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.circle(copy_frame, (700,0), 4, (255, 0, 0), 2)
            #cv2.circle(copy_frame, (xc, yc), 4, (255, 0, 0), 2)
            cv2.putText(copy_frame, f'{b},{are},{a}', (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(copy_frame,(x,y),(x1,y1),(0,255,0),2)
            cv2.imshow("frame",copy_frame)

    set1 = set(identified)
    set2 = set(id_prev)
    non_common_elements = list(set1.symmetric_difference(set2))
    mem=find_it(non_common_elements,val_prev)
    cv2.waitKey(0)
    memory.append(mem)
    if len( memory)>10:
        memory.pop(0)
    id_prev=newpd
    val_prev=newval
    #frame_path = f"{output_folder}/frame_{frame_index:04d}.png"
    #  cv2.imwrite(frame_path, frame)
    print(frame.shape)
    prevframe=frame

    size_saver=temp

    identified = []

    frame_index += 1


cam.release()
cv2.destroyAllWindows()



