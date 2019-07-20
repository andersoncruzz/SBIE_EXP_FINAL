# -*- coding: utf-8 -*-

import argparse
import os
import cv2
from keras.models import load_model
import sys
from faceDetector.faceDetector import FaceDetector
# from . import FaceDetector
import numpy as np
import json
import pickle



ALUNOS = ["Adrianeleite", "Amandacruz", "Artumirapriscila", "Brendo", "BrunaEvellyn", \
          "Emilyoliveira", "Giovanamaia", "Henrique", "Joeyramone", "Ligiabarbosa", \
          "Patrick", "Rayssamemoria", "RicardoTorres", "Shalonsouza", "Tallytarebelo", \
          "Caio", "daysonhuende", "emillyfabieli", "francicleysantos", "gabrielarruda", \
          "juliosousa", "kellenmedeiros", "milenalimadeoliveira", "nalyssarodrigues", "thiagosilvaleite", \
          "vitorvasconcelos", "viviantrindade"]

ALUNOS_ID = ["aluno1", "aluno2", "aluno3", "aluno4", "aluno5", \
             "aluno6", "aluno7", "aluno8", "aluno9", "aluno10", \
             "aluno11","aluno12", "aluno13", "aluno14", "aluno15", \
            "aluno16", "aluno17", "aluno18", "aluno19", "aluno20", \
             "aluno21", "aluno22", "aluno23", "aluno24", "aluno25", \
             "aluno26", "aluno27"]

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def padding_bounding_box(bb, img_size, padding=32):
    bounding_box = np.zeros(4, dtype=np.int32)

    bounding_box[0] = np.maximum(bb[3] - padding / 2, 0)
    bounding_box[1] = np.maximum(bb[0] - padding / 2, 0)
    bounding_box[2] = np.minimum(bb[1] + padding / 2, img_size[1])
    bounding_box[3] = np.minimum(bb[2] + padding / 2, img_size[0])

    return bounding_box


def get_emotion(net, img, coordinate):
    img_cropped = img[coordinate[0]:coordinate[2], coordinate[3]:coordinate[1]]

    img_cropped = cv2.resize(img_cropped, (110, 110))
    img_cropped = np.array(img_cropped)
    img_cropped = img_cropped.reshape(1, 110, 110, 3)
    predictions = net.predict([np.array(img_cropped)])
    index_emotion = np.argmax(predictions[0])

    return index_emotion

def get_probabilities(net, img, coordinate):
    img_cropped = img[coordinate[0]:coordinate[2], coordinate[3]:coordinate[1]]

    img_cropped = cv2.resize(img_cropped, (110, 110))
    img_cropped = np.array(img_cropped)
    img_cropped = img_cropped.reshape(1, 110, 110, 3)
    predictions = net.predict([np.array(img_cropped)])
    # index_emotion = np.argmax(predictions[0])

    return predictions[0]


def ajust_point_on_view(point, img_size):
    if point + 120 > img_size:
        return point - (point + 120 - img_size)

    return point

def get_coordinate_emoji(bb, img_size):
    bounding_box = np.zeros(4, dtype=np.int32)

    bounding_box[0] = ajust_point_on_view(bb[0], img_size[1])
    bounding_box[1] = ajust_point_on_view(bb[1], img_size[0])
    bounding_box[2] = bb[0] + 120
    bounding_box[3] = bb[1] + 120

    return bounding_box

def classify(net, img, faceDetector, img_size):
    faces = []
    # coordinates = faceDetector.detectHogSVM(img)
    coordinates = faceDetector.detectMTCNN(img)
    for coordinate in coordinates:
        face = {}
        face["bounding_box"] = padding_bounding_box(coordinate, img_size)
        face["emotion"] = get_emotion(net, img, coordinate)
        faces.append(face)

    return faces

def classify_microsoft(net, img, faceDetector, img_size):
    _to_microsoft = []
    # coordinates = faceDetector.detectHogSVM(img)
    coordinates = faceDetector.detectMTCNN(img)

    for coordinate in coordinates:
        print(coordinate)
        face = {}
        face['bounding_box'] = padding_bounding_box(coordinate, img_size)

        predictions = get_probabilities(net, img, coordinate)
        faceEmotion = {}
        faceEmotion['bounding_box'] = face['bounding_box']
        faceEmotion["emotion"] = get_emotion(net, img, coordinate)
        faceEmotion['faceRectangle'] = {"height": int(face["bounding_box"][3]),
                                        "left": int(face["bounding_box"][1]),
                                        "top": int(face["bounding_box"][0]),
                                        "width": int(face["bounding_box"][2])}
        faceEmotion['scores'] = {"anger": float(predictions[0]),
                                 "contempt": float(predictions[1]),
                                 "disgust": float(predictions[1]),
                                 "fear" : float(predictions[2]),
                                 "happiness": float(predictions[3]),
                                 "neutral": float(predictions[6]),
                                 "sadness": float(predictions[4]),
                                 "surprise": float(predictions[5])}

        #    EMOTIONS = ['angry', 'disgusted', 'fearful', \
        #             'happy', 'sad', 'surprised', 'neutral']

        _to_microsoft.append(faceEmotion)

    # print("faceEmotion: ", _to_microsoft)
    return _to_microsoft

def add_logs(user, photo, faces_):

    faces = json.dumps(faces_, cls=NumpyEncoder)
    path = "experimento_pkg/logs_SSFER_2"
    makedirs(path)
    user_1 = ALUNOS_ID[ALUNOS.index(user)]
    f = open(os.path.join(path, user_1 + ".csv"), "a+")
    # f.write(photo + "$" + json.dumps(faces, ensure_ascii=False) + "\n")
    f.write(photo + "$" + str(faces) + "\n")
    f.close()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_overlays(frame, faces, EMOTIONS, feelings_faces, img_size):
    if faces is not None:
        for face in faces:
            face_bb = face["bounding_box"]
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face["emotion"] is not None:
                # print("Person: ", face.name)
                cv2.putText(frame, EMOTIONS[face["emotion"]], (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

                emoji = feelings_faces[face["emotion"]]
                emoji_bb = get_coordinate_emoji(face_bb, img_size)
                # Ugly transparent fix
                for c in range(0, 3):
                    frame[emoji_bb[1]:emoji_bb[3], emoji_bb[0]:emoji_bb[2], c] = emoji[:,:,c] * (emoji[:, :, 3] // 255.0) +  frame[emoji_bb[1]:emoji_bb[3], emoji_bb[0]:emoji_bb[2], c] * (1.0 - emoji[:, :, 3] // 255.0)


def readFile (user):
   #/home/anderson/Projetos/exp_SBIE/logs-photos/file_photos
   with open ("/home/anderson/Projetos/exp_SBIE/logs-photos/file_photos/"+user+".txt") as fp:
      for line in fp:
         photo = line.replace("\n", "")
         # print(photo)
         # send(user, photo)

def send(user, foto):
    data = classify()
    log = open("experimento_pkg/logs-classification/"+user+".csv", "a+")
    log.write(foto +"$"  + data  + "\n")
    log.close()


def main(args):

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED)

    EMOTIONS = ['angry', 'disgusted', 'fearful', \
            'happy', 'sad', 'surprised', 'neutral']

    EMOTIONS_pt = ['raiva', 'desgosto', 'medo', \
            'felicidade', 'tristeza', 'surpresa', 'neutralidade']


    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('../emojis/' + emotion + '.png', -1))


    PATH_NET_INPUT = "/home/anderson/Projetos/SSFER/experimentos/185/VGG19+185+01_11_09+2019-07-02/weights.04-0.66.hdf5"
    PATH_SVM_INPUT = "/home/anderson/Projetos/SSFER/experimentos/185/VGG19+185+01_11_09+2019-07-02/svm.pkl"
    with open(PATH_SVM_INPUT, 'rb') as infile:
        SVM, class_names = pickle.load(infile)

    net = load_model(PATH_NET_INPUT)

    faceDetector = FaceDetector(mtcnn=True)

    users = os.listdir("/home/anderson/Projetos/exp_SBIE/logs-photos")
    for user in users:
        print(user)
        if user == "file_photos" or user == "BKP-FOTOS" or user =="gisevitor" or user == "MatheusLima"\
                or user == "BrunaEvellyn.txt" or user == "gisevitor.csv" or user == "Henrique.txt"\
                or user == "NOTLOGCLICK":
            continue

        with open("/home/anderson/Projetos/exp_SBIE/logs-photos/file_photos/" + user + ".txt") as fp:
            for line in fp:
                photo = line.replace("\n", "")
                print(photo)
                # send(user, photo)

                # Capture frame-by-frame
                # print("/home/anderson/Projetos/exp_SBIE/logs-photos/" + user + "/" + photo)
                frame = cv2.imread("/home/anderson/Projetos/exp_SBIE/logs-photos/" + user + "/" + photo)
                try:
                    img_size = np.asarray(frame.shape)[0:2]
                except AttributeError as error:
                    continue
                # faces = classify(net, frame, faceDetector, img_size)
                faces_microsoft = classify_microsoft(net, frame, faceDetector, img_size)

                add_overlays(frame, faces_microsoft, EMOTIONS_pt, feelings_faces, img_size)
                add_logs(user, photo, faces_microsoft)
                #
                # # cv2.imshow('Video', frame)
                cv2.imshow("window", frame)
                cv2.waitKey(20)
    # When everything is done, release the capture
    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
