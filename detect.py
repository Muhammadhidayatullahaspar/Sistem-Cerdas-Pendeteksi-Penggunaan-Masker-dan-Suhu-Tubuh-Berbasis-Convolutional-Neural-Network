# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from Adafruit_AMG88xx import Adafruit_AMG88xx
import math

def speak(text):
    from gtts import gTTS
    import os
    tts = gTTS(text=text, lang='id')
    tts.save("tmp_talk.mp3")
    os.system("omxplayer tmp_talk.mp3")
    os.system("rm -f tmp_talk.mp3")
sensor = Adafruit_AMG88xx()
mapAddr = [[0,1,2,3,4,5,6,7],
           [8,9,10,11,12,13,14,15],
           [16,17,18,19,20,21,22,23],
           [24,25,26,27,28,29,30,31],
           [32,33,34,35,36,37,38,39],
           [40,41,42,43,44,45,46,47],
           [48,49,50,51,52,53,54,55],
           [56,57,58,59,60,61,62,63]]
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            #sensor is an 8x8 grid so lets do a square
            height = 800
            width = 800
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = "/home/pi/facemask_detection-main/face_detector/deploy.prototxt"
weightsPath = "/home/pi/facemask_detection-main/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")

vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800, height =800)

   
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        pixels = (frame,(startX +50, startY +40))
        pixels = sensor.readPixels()
        avr_temp = np.mean(pixels)
        max_temp = np.amax(pixels)
        suhunyaa = max_temp
        for i in range (10):
            suhunya = (suhunyaa + max_temp)
            akhir = ((suhunya / 10) +31)
            last = str(akhir)
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        tanda = "." if mask > withoutMask else "."
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        if(label =="Mask") :
            speak("Terima Kasih Telah Menggunakan Masker, suhu tubuh andaa")
            speak('Suhu tubuh anda',last)
        else:
            speak("Mohon gunakan masker andaa")
            speak('Suhu tubuh anda',last)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame,"SUHU TUBUH", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
        cv2.putText(frame,str ((suhunya / 10) +31), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, label,(startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, tanda, (startX +60, startY + 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
        cv2.rectangle(frame,  (startX, startY), (endX, endY), color, 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
