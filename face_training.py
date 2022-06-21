import cv2 as cv
import numpy as np
from PIL import Image
import os

def get_images_and_labels(path, detector):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        PIL_image = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_image, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return face_samples,ids


path='dataset'
recognizer=cv.face.LBPHFaceRecognizer_create()
detector=cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

faces,ids = get_images_and_labels(path, detector)
print("\n [INFO] Training faces. This will take a bit ...")
recognizer.train(faces, np.array(ids))
recognizer.write('./trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting".format(len(np.unique(ids))))
