import torch
import numpy as np
import pandas as pd
import cv2
from time import time
import sys
import threading
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import VideoStream


class ObjectDetection:
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """

    def __init__(self, input_file = 0, out_file="Labeled_Video.mp4"):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.input_file = input_file
        self.model = self.load_model()
        self.model.conf = 0.001 # set inference threshold at 0.3
        self.model.iou = 0.65 # set inference IOU threshold at 0.3
        #print(self.model.classes)
        self.model.classes = [0, 63] # set model to only detect "Person" and laptop class
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_300p(self, cap):
        cap.set(3, 300)
        cap.set(4, 300)
        return

    def make_480p(self, cap):
        cap.set(3, 640)
        cap.set(4, 480)
        return

    def make_240p(self, cap):
        cap.set(3, 320)
        cap.set(4, 240)
        return

    def get_video_from_camera(self, input_file):
        """
        Function creates a streaming object to read the video from the file frame by frame.
        :param self:  class object
        :return:  OpenCV/imutils.VideoStream object to stream video frame by frame.
        """

        #cap = WebcamVideoStream(src=input_file, resolution=(300,300)).start()
        cap = VideoStream(src=input_file, resolution=(300,300)).start()

        #cap = cv2.VideoCapture(input_file, cv2.CAP_ANY)

        #self.make_300p(cap)
        assert cap is not None
        return cap 

    def load_model(self):
        """
        Function loads the detection model from PyTorch Hub.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Loads YOLOv5s from torch hub
        
        #model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', map_location = torch.device('cpu'))
        #model.to('cpu')
        #print(model.classes)
        return model

    def score_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        self.model.to(self.device)
        results = self.model([frame])
        #print(results.pandas().xyxyn[0])
        labels, cord, classNo = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy(), results.xyxyn[0][:, -1].to('cpu').numpy()
        #print(labels)
        return labels, cord, classNo

    def plot_boxes(self, results, frame, count, bcount):
        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        labels, cord, classNo = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 0, 255)
            if(row[4]*100 >= 75 and labels[i] == 0): # Only plots box around target if recognition is over 75% confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1) # Plots rectangle around target
                label = f"{int(row[4]*100)}" # Labels the target with the confidence of the recognition
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1) # Adds the label to the target
                cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Displays total targets
                count+= 1
            elif(row[4]*100 >= 75 and labels[i] == 63): # Only plots box around target if recognition is over 75% confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1) # Plots rectangle around target
                label = f"{int(row[4]*100)}" # Labels the target with the confidence of the recognition
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1) # Adds the label to the target
                cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Displays total targets
                bcount+= 1
        return frame, count, bcount

    def scan_and_plot(self, frame):
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)
        return frame

    def __call__(self):
        print(self.input_file)
        #players = self.get_video_from_camera() # create streaming service for application
        player = self.get_video_from_camera(0)
        #self.make_300p(player)
        player2 = self.get_video_from_camera(1)
        #player = players[0]
        #player2 = players[1]
        #assert player.isOpened() and player2.isOpened()
        assert player is not None
        #x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        #y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_shape = 300
        y_shape = 300
        
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 15, (x_shape*2, y_shape*2))
        fc, fps, tfcc = 0, 0, 0
        #fps = 0
        tfc = 1
        #tfc = int(player.get(cv2.CAP_PROP_FRAME_COUNT))
        #tfcc = 0
        count = 1
        bcount= 1
        camno = 0
        while True:
            fc += 1
            start_time = time()
            if (count%5)==0 and camno == 0:
              #  ret, frame = player2.read()
                frame = player2.read()
                print("Switched to Camera 2")
                camno =1
                count+=1
            elif (bcount%5)==0 and camno == 1:
               # ret, frame = player2.read()
                frame = player.read()
                camno = 0
                bcount+=1
                print("Switched to Camera 1")
            elif(camno == 1):
                frame = player2.read()
            else:
               # ret, frame = player.read()
                frame = player.read()
                camno = 0
            # if not ret:
            #    break
            #print(count)
            #print(bcount)
            #frame = imutils.resize(frame, width=300)
            results = self.score_frame(frame)
            frame, count, bcount = self.plot_boxes(results, frame, count, bcount)
            #frame = self.scan_and_plot(frame)
            #cv2.imshow('Frame', frame)
            end_time = time()
            fps += 1/(end_time - start_time +0.001)
            if (fc == 10 and fps != 0):
                fps = (fps / 10)
                tfcc += fc
                fc = 0
                per_com = int(tfcc / tfc * 100)
                print(f"Frames Per Second : {fps} || Percentage Parsed : {per_com}")
            (B, G, R) = cv2.split(frame)
            #zeros = np.zeros((y_shape, x_shape), dtype= "uint8")
            #R = R.astype(zeros.dtype)
            #R = cv2.merge([zeros, zeros, R])
            #G = cv2.merge([zeros, G, zeros])
            #B = cv2.merge([B, zeros, zeros])
            output = np.zeros((y_shape*2, x_shape*2, 3), dtype="uint8")
            #output[0:y_shape, 0:x_shape] = frame
            #output[0:y_shape, x_shape:x_shape * 2] = R
            #output[y_shape:y_shape * 2, x_shape:x_shape * 2] = G
            #output[y_shape:x_shape * 2, 0:x_shape] = B
            output = cv2.merge((B, G, R))
            out.write(output)
            cv2.imshow('Frame', output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):break
        player.stop()
        player2.stop()
        out.release()


link = sys.argv[1]
output_file = sys.argv[2]
a = ObjectDetection(link, output_file)
a()