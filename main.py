import cv2
from ultralytics import YOLO
import os
import numpy as np
import concurrent.futures

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture('media1.mp4')

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
                #print("filtered_rectangles:",filtered_rectangles)

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
def overlap(prev,frame):

    kernel = np.array((9, 9), dtype=np.uint8)
    frame1=prev
    frame2=frame
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

    # Create a copy of the original frame for drawing contours
    frame_with_contours = frame2.copy()
    
    contours1=combine_close_rectangles(contours)
    contours1=avoid_overlap(contours1)

    length=len(contours1)
    for x, y, w,h ,a in contours1:#c:close to camera or not
        xc=x+int(w/2)
        yc=y+int(h/2)
        x1=x + w
        y1=y + h
        cv2.circle(frame_with_contours,(xc,yc),4,(255,0,0),2)
        text = f"({xc}, {yc},area:,{w*h},{a})"
        cv2.putText(frame_with_contours, text, (xc + 10, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame_with_contours, (x, y), (x1, y1), (0, 255, 0), 2)

    # Display the original frame with highlighted contours using OpenCV
    cv2.putText(frame_with_contours, f'number_of_birds:{length}', (20+ 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Frame with Contours", frame_with_contours)
    cv2.waitKey(0)




frame_index = 0



prevframe=0
while True:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    ret, frame = cam.read()

    if not ret:
        break  # Break the loop if no more frames are available


    if frame_index == 0:
        prevframe=frame
        frame_index += 1
        continue  # Skip the first frame
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        overs = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # each vechicle box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

    detection=overlap(prevframe,frame)


    prevframe=frame



    frame_index += 1

cam.release()
cv2.destroyAllWindows()



