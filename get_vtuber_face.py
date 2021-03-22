import cv2
import glob
import os.path

CASCADE_FILE = './cascade/lbpcascade_animeface.xml'
# FACE_ASPECT_SIZE = (128, 128)
FACE_ASPECT_SIZE = (64, 64)

def get_face(file_path, file_name):

    # 画像をグレースケールへ返還
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 顔部分をキャプチャ
    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))
    
    # キャプチャ部分すべてをイメージ化
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_ASPECT_SIZE)
        cv2.waitKey(0)
        cv2.imwrite('{0}{1}.jpg'.format('./img/training_images-64/', str(file_name)), face)


if __name__ == "__main__":
    file_path_list = glob.glob('./img/fixed_images/*')
    
    for i, file_path in enumerate(file_path_list):
        get_face(file_path, i)