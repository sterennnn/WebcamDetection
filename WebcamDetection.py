import torch
import numpy as np
import pandas as pd
import cv2
from time import time
import sys
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import VideoStream

class ObjectDetection:
    """
    The class performs generic object detection on a webcam or connected camera.
    It uses yolov5s pretrained model to make inferences and uses OpenCV to import and read frames to run the detection on.
    Included Features:
    1. Reading video input from a camera using OpenCV/imutils
    2. Writing output to video file using OpenCV !! Not Working !!
    3. Using pretrained model to make inferences on frames.
    4. Use the inferences to plot boxes on objects and labels around the detected targets.
    """

    def __init__(self, input_num = 0, out_file="Labeled_Video.mp4"):
        """
        :param input_file: provide a camera input number from the system to use as input video feed.
        :param out_file: name of a existing file, or a new file in which to write the output video to.
        :return: void
        """
        self.input_num = input_num
        self.model = self.load_model()
        self.model.conf = 0.001 # set inference threshold at 0.3
        self.model.iou = 0.65 # set inference IOU threshold at 0.3
        #print(self.model.classes)
        self.model.classes = [0, 63] # set model to only detect "Person" and laptop class
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simple resolution changing functions for OpenCV

    def make_300p(self, cap):
        cap.set(3, 300)
        cap.set(4, 300)
        return

    def make_480p(self, cap):
        cap.set(3, 640)
        cap.set(4, 480)
        return

    def make_640p(self, cap):
        cap.set(3, 640)
        cap.set(4, 640)
        return



    def get_video_from_camera(self, input_num):
        """
        Function creates a streaming object to read the video from the camera input frame by frame.
        :param self:  class object, input camera number
        :return:  OpenCV/imutils.VideoStream object to stream video to be read frame by frame.
        """

        #cap = WebcamVideoStream(src=input_file).start()
        #cap = VideoStream(src=input_file).start() #, resolution=(300,300)
        cap = VideoStream(src=input_num, width=640, height=640).start()

        #cap = cv2.VideoCapture(input_file, cv2.CAP_ANY)

       # self.make_640p(cap)
        assert cap is not None
        return cap 

    def load_model(self):
        """
        Function loads the detection model from PyTorch Hub.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True) # Loads YOLOv5s from torch hub
        
        #model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', map_location = torch.device('cpu'))
        #model.to('cpu')
        #print(model.classes)
        return model

    def score_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to make inferences from.
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
        print(self.input_num)
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
        x_shape = 640
        y_shape = 640
        
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 15, (x_shape, y_shape))
        fc, fps, tfcc = 0, 0, 0
        #fps = 0
        tfc = 1
        #tfc = int(player.get(cv2.CAP_PROP_FRAME_COUNT))
        #tfcc = 0
        count = 1
        bcount= 1
        camno = 0
        print("Press Spacebar to move to the next perspective, Escape to move to the previous perspective and Q to exit")
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
            frame = imutils.resize(frame, width=640)
            
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
            output = np.zeros((y_shape, x_shape, 3), dtype="uint8")
            output = cv2.merge(np.uint8((B, G, R)))
            #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            #outputShow = imutils.resize(output, width=600)
            out.write((output).astype("uint8"))#.cv2.cvtColor(cv2.COLOR_BGR2RGB))
            cv2.imshow('Frame', output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):break
            if key == 32: camno+=1
            if key == 27: camno-=1
        player.stop()
        player2.stop()
        out.release()


link = sys.argv[1]
output_file = sys.argv[2]
a = ObjectDetection(link, output_file)
a()