import face_recognition
import cv2
import numpy as np
import os

# Viola Jones:  [49520, 45794, 0, 0]
# HOG SVM:  [54790, 40524, 0, 0]
# CNN:  [79966, 15348, 0, 0]

#DATABASE FER
# Viola Jones:  [21855, 73459, 0, 0]
# HOG SVM:  [54790, 40524, 0, 0]
# CNN:  [79966, 15348, 0, 0]

#VOC2007
# 4952/4952 100.00%
# Viola Jones:  [507, 2905, 1500, 40]
# HOG SVM:  [711, 2915, 1296, 30]
# CNN:  [829, 2932, 1178, 13]



#TODO: To add MTCNN DETECTOR NEURAL NETWORK

class FaceDetector:
    def __init__(self, mtcnn=False):
        if mtcnn == True:
            from mtcnn.mtcnn import MTCNN
            self.mtcnn = MTCNN()

    def detectViolaJones(self, image, SF=1.3, casc_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "haarcascade_file.xml")):
        #TODO: Verify params ViolaJones, IS REALLY VIOLA JONES ALGORITHM?
        cascade_classifier = cv2.CascadeClassifier(casc_path)
        faces = cascade_classifier.detectMultiScale(
            image,
            scaleFactor = SF,
            minNeighbors = 5
        )
        # None is we don't found an image
        if not len(faces) > 0:
            #print ("No hay caras")
            return []

        max_area_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                max_area_face = face
            # Chop image to face
            face = max_area_face
            #print(face)
            #image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
        return faces

    def detectHogSVM(self, img):
        return face_recognition.face_locations(img)

    def detectCNN(self, img, upsample=1):
        if img.size == 0:
            print("[+++] Entrou no None")
            return []

        return face_recognition.face_locations(img, number_of_times_to_upsample=upsample, model="cnn")

    def detectMTCNN(self, img):
        result = self.mtcnn.detect_faces(img)

        if result == []:
            return []
        bbs = []
        for rt in result:
            # if rt["confidence"] < 0.1:
            #     continue

            bb = rt["box"]
            bb = [0 if x < 0 else x for x in list(bb)]
            #bb[1], bb[2], bb[3], bb[0]
            bbs.append((bb[1],  bb[2] + bb[0], bb[3] + bb[1], bb[0]))

        return bbs
