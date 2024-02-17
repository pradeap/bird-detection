import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QHBoxLayout
from PyQt5.QtWidgets import QGraphicsBlurEffect, QGraphicsOpacityEffect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np

class CCTVApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCTV Application")
        self.setGeometry(500, 500, 1000, 500)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create labels for video streams
        self.video_label1 = QLabel(self)
        self.video_label1.setAlignment(Qt.AlignCenter)



        # Create central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)



        self.video_label2 = QLabel(self)
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.count_label1 = QLabel("Detections: 0", self)
        self.count_label1.setAlignment(Qt.AlignCenter)

        self.count_label2 = QLabel("Detections: 0", self)
        self.count_label2.setAlignment(Qt.AlignCenter)

        # Create a start button
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_video_capture)

        # Create a layout for labels and button
        video_layout1 = QVBoxLayout()
        video_layout1.addWidget(self.video_label1)
        video_layout1.addWidget(self.count_label1)

        video_layout2 = QVBoxLayout()
        video_layout2.addWidget(self.video_label2)
        video_layout2.addWidget(self.count_label2)

        # Combine video layouts and add the start button
        self.layout = QHBoxLayout()
        self.layout.addLayout(video_layout1)
        self.layout.addLayout(video_layout2)
        self.layout.addWidget(self.start_button)


        self.central_widget.setLayout(self.layout)

        # Set the paths to your saved video files
        self.video_path1 = 'media.mp4'
        self.video_path2 = 'media.mp4'

        self.video_capture1 = cv2.VideoCapture(self.video_path1)
        self.video_capture2 = cv2.VideoCapture(self.video_path2)

        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(lambda: self.update_frame(1))

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(lambda: self.update_frame(2))

        # Initialize variables for video analysis
        self.frame_index1 = 0
        self.prev_frame1 = None
        self.id_counter1 = 1

        self.frame_index2 = 0
        self.prev_frame2 = None
        self.id_counter2 = 1

    def start_video_capture(self):
        if not self.timer1.isActive() and not self.timer2.isActive():
            self.timer1.start(40)  # Update every 30 milliseconds for video 1
            self.timer2.start(40)  # Update every 30 milliseconds for video 2
            self.start_button.setText("Stop")
        else:
            self.timer1.stop()
            self.timer2.stop()
            self.start_button.setText("Start")

    def update_frame(self, video_number):
        if video_number == 1:
            ret, frame = self.video_capture1.read()
            label = self.video_label1
            prev_frame = self.prev_frame1
            id_counter = self.id_counter1
        elif video_number == 2:
            ret, frame = self.video_capture2.read()
            label = self.video_label2
            prev_frame = self.prev_frame2
            id_counter = self.id_counter2
        else:
            return

        if ret:
            if video_number == 1:
                frame_index = self.frame_index1
            elif video_number == 2:
                frame_index = self.frame_index2
            else:
                return

            if frame_index == 0:
                prev_frame = frame.copy()
                frame_index += 1
            detection = self.overlap(prev_frame, frame)
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            count=0
            for x, y, w, h, a in detection:
                xc = x + int(w / 2)
                yc = y + int(h / 2)
                x1 = x + w
                y1 = y + h
                text = f"{id_counter}"
                cv2.circle(frame_copy, (xc, yc), 4, (255, 0, 0), 2)
                cv2.putText(frame_copy, text, (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.rectangle(frame_copy, (x, y), (x1, y1), (0, 255, 0), 2)

                id_counter += 1
                count+=1
            frame_copy = cv2.resize(frame_copy, (320, 240))  # Adjust the size as needed
            h, w, ch = frame_copy.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)

            if video_number == 1:
                self.prev_frame1 = frame.copy()
                self.frame_index1 += 1
                self.id_counter1 = id_counter
                self.count_label1.setText(f"Detections: {count}")
            elif video_number == 2:
                self.prev_frame2 = frame.copy()
                self.frame_index2 += 1
                self.id_counter2 = id_counter
                self.count_label2.setText(f"Detections: {count}")

    def filter(self,image):

        filtered_image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return filtered_image

    def re_combine_close_rectangles(self,contours, threshold_distance=60):  # closest objects from camera
        combined_rectangles = []

        for i, contour in enumerate(contours):
            x1, y1, w1, h1 = cv2.boundingRect(contour)

            # Check against already combined rectangles
            merged = False
            for j, (x2, y2, w2, h2) in enumerate(combined_rectangles):
                # Calculate the distance between the centers of bounding rectangles
                distance = np.sqrt((x1 + w1 / 2 - x2 - w2 / 2) ** 2 + (y1 + h1 / 2 - y2 - h2 / 2) ** 2)

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

        combined_rectangles = [(x, y, w, h, 0) for x, y, w, h in combined_rectangles if
                               w >= 10 and h >= 10 and w * h >= 500]

        return combined_rectangles

    def combine_close_rectangles(self,contours, threshold_scale=0.9):

        combined_rectangles = []
        temp = []
        for i, contour in enumerate(contours):
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            if y1 + int(h1 / 2) > 750:  # boundary decider for closest object

                temp.append(contour)

            else:
                area1 = w1 * h1

                # Check against already combined rectangles
                merged = False

                for j, (x2, y2, w2, h2, a) in enumerate(combined_rectangles):
                    a = 2
                    area2 = w2 * h2
                    threshold_distance1 = threshold_scale * max(w1, h1, w2, h2)
                    threshold_distance2 = threshold_scale * np.sqrt((area1 + area2) / 2)
                    distance = np.sqrt((x1 + w1 / 2 - x2 - w2 / 2) ** 2 + (y1 + h1 / 2 - y2 - h2 / 2) ** 2)
                    threshold_distance = max(threshold_distance1, threshold_distance2)
                    if distance < threshold_distance:
                        w = max(x1 + w1, x2 + w2) - min(x1, x2)
                        h = max(y1 + h1, y2 + h2) - min(y1, y2)
                        if w * h >= 1000:
                            a = 1
                        combined_rectangles[j] = (
                            min(x1, x2), min(y1, y2),
                            w, h, a
                        )

                        merged = True
                        break

                if not merged:
                    # If not close to any existing rectangle, add it as a new rectangle
                    combined_rectangles.append((x1, y1, w1, h1, 2))
        temp = self.re_combine_close_rectangles(temp)
        combined_rectangles = combined_rectangles + temp
        # combined_rectangles1=calculate_iou(combined_rectangles) #Avoid overlap tech
        return combined_rectangles

    def calculate_iou1(self,box1, box2):
        x11, y11, x21, y21, a1 = box1
        x12, y12, x22, y22, a2 = box2
        x1_int = max(box1[0], box2[0])
        y1_int = max(box1[1], box2[1])
        x2_int = min(box1[2], box2[2])
        y2_int = min(box1[3], box2[3])
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)

        merge_area = (x2_int - x1_int) * (y2_int - y1_int)
        if ((x12 <= x11 <= x22) and (y12 <= y11 <= y22)) or ((x12 <= x21 <= x22) and (y12 <= y21 <= y22)) or (
                (x11 <= x12 <= x21) and (y11 <= y12 <= y21)) or ((x11 <= x22 <= x21) and (y11 <= y22 <= y21)):
            if box1_area >= box2_area:
                return ((box1_area - merge_area) / box1_area, 1)
            else:
                return ((box2_area - merge_area) / box2_area, 2)
        else:
            return 0, 0

    def filter_overlapping_rectangles(self,rectangles, overlap_threshold=0.2):
        filtered_rectangles = []

        for rect1 in rectangles:
            if len(filtered_rectangles) == 0:
                filtered_rectangles.append(rect1)
            else:
                temp = []
                t = 0
                p = 0
                for rect2 in filtered_rectangles:
                    # print("filtered_rectangles:",filtered_rectangles)

                    iou, pos = self.calculate_iou1(rect1, rect2)

                    if pos == 0:
                        temp.append(rect2)
                        t = 1

                    if pos == 1 and iou > 0:
                        p = 1
                        temp.append(rect1)
                    if pos == 2 and iou > 0:
                        p = 1
                        temp.append(rect2)

                if t == 1 and p == 0:
                    temp.append(rect1)

                filtered_rectangles = temp
        return filtered_rectangles

    def calculate_distance(self,rect1, rect2):
        # Assuming rectangles are represented as (x, y, x1, y1, area, id)
        x1, y1, _, _, _ = rect1
        x2, y2, _, _, _ = rect2

        # Calculate Euclidean distance between rectangle centers
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance

    def group_rectangles(self,rectangles, threshold):
        groups = []
        assigned_indices = set()

        for i, rect1 in enumerate(rectangles):
            if i not in assigned_indices:
                group = [i]
                for j, rect2 in enumerate(rectangles):
                    if i != j and j not in assigned_indices:
                        distance = self.calculate_distance(rect1, rect2)
                        if distance < threshold:
                            group.append(j)
                            assigned_indices.add(j)

                groups.append(group)

        return groups

    def side_overlap(self,contours):
        contours1 = []
        threshold = 10
        groups = self.group_rectangles(contours, threshold)
        for i, val in enumerate(groups):
            x = 100000
            x1 = 0
            y = 100000
            y1 = 0
            a = 0
            if len(val) >= 2:
                for j in val:
                    x = min(x, contours[j][0])
                    y = min(y, contours[j][1])
                    x1 = max(x1, contours[j][2])
                    y1 = max(y1, contours[j][3])
                    a = max(a, contours[j][4])
                contours1.append((x, y, x1, y1, a))



            else:
                val = val[0]
                contours1.append(contours[val])
        return contours1

    def avoid_overlap(self,contours):
        contours1 = []
        for x, y, w, h, a in contours:
            x1 = x + w
            y1 = y + h
            contours1.append((x, y, x1, y1, a))
        filtered_rectangles = self.filter_overlapping_rectangles(contours1, overlap_threshold=0.4)
        filtered_rectangles = self.side_overlap(filtered_rectangles)
        filtered_rectangles1 = []

        for x, y, x1, y1, a in filtered_rectangles:
            w = x1 - x
            h = y1 - y
            filtered_rectangles1.append((x, y, w, h, a))

        return filtered_rectangles1

    # def display(contours):

    def overlap(self,prev, frame):
        kernel = np.array((9, 9), dtype=np.uint8)
        frame1 = prev
        frame2 = frame
        img1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        g_frame1 = img1
        g_frame2 = img2
        grayscale_diff = cv2.subtract(g_frame1, g_frame2)
        frame_diff = cv2.medianBlur(grayscale_diff, 3)
        mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        # Create a copy of the original frame for drawing contours
        frame_with_contours = frame2.copy()

        contours = self.combine_close_rectangles(contours)
        contours = self.avoid_overlap(contours)
        return contours

if __name__ == '__main__':
    app = QApplication(sys.argv)
    cctv_app = CCTVApp()
    cctv_app.show()
    sys.exit(app.exec_())
