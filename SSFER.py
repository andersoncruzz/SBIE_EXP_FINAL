import pickle
from architectures import *
from keras import backend as K
import numpy as np
from faceDetector.faceDetector import FaceDetector
import cv2

from utils import padding_bounding_box

class SSFER:
    def __init__(self,
                 CNN_ARCHITECTURE="VGG19",
                 SIZE=185,
                 CNN_PATH="/home/anderson/Projetos/SSFER/experimentos/185/VGG19+185+01_11_09+2019-07-02/weights.04-0.66.hdf5",
                 CLASSIFIER_PATH="/home/anderson/Projetos/SSFER/experimentos/185/VGG19+185+01_11_09+2019-07-02/svm.pkl"):

        self.SIZE = SIZE
        self.ARCHITECTURE = CNN_ARCHITECTURE
        self.CNN_PATH = CNN_PATH
        self.CLASSIFIER_PATH = CLASSIFIER_PATH

        if CNN_ARCHITECTURE == "VGG19":
            self.net = VGG19()
        elif CNN_ARCHITECTURE == "MobileNetV2":
            self.net = MobileNetV2()
        elif CNN_ARCHITECTURE == "ResNet50":
            self.net = ResNet50()
        elif CNN_ARCHITECTURE == "InceptionV3":
            self.net = InceptionV3()
        elif CNN_ARCHITECTURE == "InceptionResNetV2":
            self.net = InceptionResNetV2()


        self.net = self.net.build_network((SIZE, SIZE, 3), 7)
        self.net.load_weights(CNN_PATH)

        with open(CLASSIFIER_PATH, 'rb') as infile:
            self.classifier, self.classes = pickle.load(infile)

        self.faceDetector = FaceDetector(mtcnn=False)
        self.detector_name = "MMOD-CNN"

    def detect_faces(self, img):
        # coordinates = self.faceDetector.detectHogSVM(img)
        # coordinates = self.faceDetector.detectMTCNN(img)


        coordinates = self.faceDetector.detectCNN(img)
        return coordinates


    def cropping_face(self, coordinate, img):
        #largura 640
        #altura 480

        print("coordinate: ", coordinate)
        #SVM [(241, 455, 480, 134)]

        #MTCNN
        # img_cropped = img[coordinate[0]:coordinate[2], coordinate[3]:coordinate[1]]

        # img[y:y + h, x:x + w]

        #CNN [(212, 457, 480, 105)]
        # print("y:", coordinate[3], "y+h:", coordinate[1], "x:", coordinate[0], "x+w:", coordinate[2])
        img_cropped = img[coordinate[3]:coordinate[1], coordinate[0]:coordinate[2]]

        img_cropped = cv2.resize(img_cropped, (self.SIZE, self.SIZE))
        img_cropped = np.array(img_cropped)
        img_cropped = img_cropped.reshape(1, self.SIZE, self.SIZE, 3)
        return img_cropped

    def extract_features(self, data):
        last_output = K.function([self.net.layers[0].input], [self.net.layers[-2].output])
        # new_X = []

        # for i in range(len(data)):
        img = np.array(data)
        img = img.astype('float32')
        img = img.reshape(1, self.SIZE, self.SIZE, 3)

        features = last_output([img])
            # new_X.append(features[0][0])

        return features[0][0]

    def classify_probabilities(self, features):
        probabilities = self.classifier.predict([features])
        return self.classes[probabilities[0].item()], probabilities[0].item()

    def classify(self, img, image_original_size):
        print("DETECTING FACES")
        coordinates = self.detect_faces(img)
        faces = []

        for coordinate in coordinates:
            face_dict = {}
            print("CROPPING")
            face = self.cropping_face(coordinate, img)
            print("FEATURES")
            features = self.extract_features(face)
            print("CLASSIFING")
            emotion, index_emotion = self.classify_probabilities(features)

            face_dict["bounding_box"] = padding_bounding_box(coordinate, image_original_size)
            face_dict["emotion"] = emotion
            face_dict["index_emotion"] = index_emotion

            if self.detector_name == "MMOD-CNN":
                print("ENTROU NO IF")
                print(face_dict["bounding_box"])

                face_dict["bounding_box"] = [face_dict["bounding_box"][0].item(),
                                             face_dict["bounding_box"][1].item(),
                                             face_dict["bounding_box"][2].item(),
                                             face_dict["bounding_box"][3].item()]

                print(face_dict["bounding_box"])

                face_dict["faceRectangle"] = {"height": face_dict["bounding_box"][3],
                                              "left": face_dict["bounding_box"][1],
                                              "top": face_dict["bounding_box"][0],
                                              "width": face_dict["bounding_box"][2]}

            else:
                face_dict["faceRectangle"] = {"height": face_dict["bounding_box"][3],
                                                "left": face_dict["bounding_box"][1],
                                                "top": face_dict["bounding_box"][0],
                                                "width": face_dict["bounding_box"][2]}

            # face_dict['scores'] = {"anger": float(predictions[0]),
            #                          "contempt": float(predictions[1]),
            #                          "disgust": float(predictions[1]),
            #                          "fear" : float(predictions[2]),
            #                          "happiness": float(predictions[3]),
            #                          "neutral": float(predictions[6]),
            #                          "sadness": float(predictions[4]),
            #                          "surprise": float(predictions[5])}

            faces.append(face_dict)
        return faces