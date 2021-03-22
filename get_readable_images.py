import os
import cv2
import glob
import mimetypes
import shutil

# 修正対象となるフォルダ
file_path_list = glob.glob('./img/original_images/**/*')

# 保存先フォルダ
SAVE_DIR = './img/fixed_images/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for i,file_path in enumerate(file_path_list):

    if mimetypes.guess_type(file_path)[0] == None:
        shutil.copyfile(file_path, SAVE_DIR+'{}{}'.format(str(i).zfill(8),'.jpg'))
    else:
        ext = os.path.splitext(file_path)[1]
        shutil.copyfile(file_path, SAVE_DIR+'{}{}'.format(str(i).zfill(8),ext))


file_path_list = glob.glob(SAVE_DIR+'*')
for file_path in file_path_list:
    print(file_path)
    image = cv2.imread(file_path)
    if image is None:
        os.remove(file_path)