import cv2
import os
import numpy as np
import json
from SSFER import SSFER
from utils import makedirs

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


def add_logs(user, photo, faces_):
    faces = json.dumps(faces_, cls=NumpyEncoder)
    path = "exp_emotions/logs_SSFER"
    makedirs(path)
    user_1 = ALUNOS_ID[ALUNOS.index(user)]
    f = open(os.path.join(path, user_1 + ".csv"), "a+")
    # f.write(photo + "$" + json.dumps(faces, ensure_ascii=False) + "\n")
    f.write(photo + "$" + str(faces) + "\n")
    f.close()


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


def add_overlays(frame, faces, EMOTIONS, feelings_faces, img_size):
    if faces is not None:
        for face in faces:
            face_bb = face["bounding_box"]
            print(face_bb)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face["emotion"] is not None:
                # print("Person: ", face.name)
                cv2.putText(frame, face["emotion"], (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

                emoji = feelings_faces[face["index_emotion"]]
                emoji_bb = get_coordinate_emoji(face_bb, img_size)
                # Ugly transparent fix
                for c in range(0, 3):
                    frame[emoji_bb[1]:emoji_bb[3], emoji_bb[0]:emoji_bb[2], c] = emoji[:,:,c] * (emoji[:, :, 3] // 255.0) +  frame[emoji_bb[1]:emoji_bb[3], emoji_bb[0]:emoji_bb[2], c] * (1.0 - emoji[:, :, 3] // 255.0)



def main():
    PATH_ROOT_SBIE = "/media/anderson/DATA/Projetos/exp_SBIE/"

    ssFER = SSFER()

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED)

    EMOTIONS = ['angry', 'disgusted', 'fearful', \
            'happy', 'sad', 'surprised', 'neutral']

    EMOTIONS_pt = ['raiva', 'desgosto', 'medo', \
            'felicidade', 'tristeza', 'surpresa', 'neutralidade']


    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

    users = os.listdir(os.path.join(PATH_ROOT_SBIE, "logs-photos"))
    for user in users:
        print(user)
        if user == "file_photos" or user == "BKP-FOTOS" or user =="gisevitor" or user == "MatheusLima"\
                or user == "BrunaEvellyn.txt" or user == "gisevitor.csv" or user == "Henrique.txt"\
                or user == "NOTLOGCLICK":
            continue

        with open(os.path.join(PATH_ROOT_SBIE, "logs-photos", "file_photos", user + ".txt")) as fp:
            for line in fp:
                photo = line.replace("\n", "")
                print(photo)
                # send(user, photo)

                # Capture frame-by-frame
                # print("/home/anderson/Projetos/exp_SBIE/logs-photos/" + user + "/" + photo)
                frame = cv2.imread(os.path.join(PATH_ROOT_SBIE, "logs-photos", user, photo))
                try:
                    img_original_size = np.asarray(frame.shape)[0:2]
                except AttributeError as error:
                    continue

                faces = ssFER.classify(frame, img_original_size)
                print("faces:", faces)
                add_overlays(frame, faces, EMOTIONS_pt, feelings_faces, img_original_size)
                add_logs(user, photo, faces)
                #
                # # cv2.imshow('Video', frame)
                cv2.imshow("window", frame)
                cv2.waitKey(10)
        break
    # When everything is done, release the capture
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
