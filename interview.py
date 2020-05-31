from tkinter import *
from tkinter import messagebox
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

def interview():

    window7 = Tk()
    window7.title("Interview")
    w=window7.winfo_screenwidth()
    h=window7.winfo_screenheight()
    resolution=str(w)+'x'+str(h)
    window7.geometry(resolution)
    #window7.resizable(False,False)
    window7.configure()
    #fram= Frame(window7)
    lbl= Label(window7, text= "Ques 1. ______________________________") #question1
    lbl.place(x= 50, y= 50)
    def video():
        happyy=0
        emotion_model_path = 'C:/Users/91996/Emotion-recognition/traning_logs/_mini_XCEPTION.45-0.72.hdf5'

        # hyper-parameters for bounding boxes shape
        # loading models
        face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        emotion_classifier = load_model(emotion_model_path, compile=True)
        EMOTIONS = ["angry", "happy", "sad", "surprised","neutral"]



        # starting video streaming
        cv2.namedWindow('your_face')
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            #frame = camera.read()[1]
            #reading the frame
            frame = imutils.resize(frame,width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frameClone = frame.copy()
            if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                            # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
                    # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)


                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                #print(label)
                d[label]=d.get(label,0)+1



            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                        # construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        w = int(prob * 300)
                        cv2.rectangle(canvas, (7, (i * 35) + 5),
                        (w, (i * 35) + 35), (0, 0, 255), -1)
                        cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
                        cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                      (0, 0, 255), 2)

            cv2.imshow('your_face', frameClone)
            cv2.imshow("Probabilities", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return happyy
    def submit():
        messagebox.showinfo("Instructions", "You are about to proceed to the video interview.\nTo close the session press 'Q' Key.")
        count=video()
        #print(d)

        

    d={}
    l1=[]
    l2=[]
    sub1= Button(window7, text= "Submit", command= submit)
    sub1.place(x= 50, y= 100)
    #ans1= Entry(window7, width= 100)
    #ans1.place(x= 50, y= 100)
    lbl= Label(window7, text= "Ques 2. ______________________________") #question2
    lbl.place(x= 50, y= 200)
    sub2= Button(window7, text= "Submit", command= submit)
    sub2.place(x= 50, y= 250)
    #ans2= Entry(window7, width= 100)
    #ans2.place(x= 50, y= 250)
    lbl= Label(window7, text= "Ques 3. ______________________________") #question3
    lbl.place(x= 50, y= 350)
    sub3= Button(window7, text= "Submit", command= submit)
    sub3.place(x= 50, y= 400)

    #ans3= Entry(window7, width= 100)
    #ans3.place(x= 50, y= 400)


    #sub= Button(window7, text= "Submit", command= submit)
    #sub.place(x= 50, y= 500)
    window7.mainloop()
    for k,v in d.items():
        temp=(v,k)
        l1.append(temp)
    l2=sorted(l1, reverse=True)
    #print(l2)
    l2=l2[0]
    return(l2[1])
    