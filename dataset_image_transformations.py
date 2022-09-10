# realdata -> mirror
# simdata -> rotate x*90°

import os
import cv2

# directorys and datasets
# ONLY RUN ONCE BECAUSE IT SAVES DIRECTLY IN THE LOADING FOLDER
datasets_dir = 'C:/Users/mam-pm/Desktop/'
dataset = '03 depthmaps_sim_grey_2000_real_1769_flip/'
output_dir = 'C:/Users/mam-pm/PycharmProjects/cycleGAN_grey/venv/data/'

for folder in os.listdir(datasets_dir + dataset):
    for image in os.listdir(datasets_dir + dataset + folder):
        if (folder == "Sim"):

            new_img = cv2.flip(cv2.imread(os.path.join(datasets_dir, dataset, folder, image), cv2.IMREAD_GRAYSCALE), 1)
            savepath = os.path.join(datasets_dir, dataset, folder, "flip_" + image)
            print(savepath)
            cv2.imwrite(savepath, new_img)
            '''
        #elif (folder == "Sim"):
            new_img = cv2.rotate(cv2.imread(os.path.join(datasets_dir, dataset, folder, image), cv2.IMREAD_GRAYSCALE),
                                 0)
            savepath = os.path.join(datasets_dir, dataset, folder, "rot90_" + image)
            print(savepath)
            cv2.imwrite(savepath, new_img)
            new_img = cv2.rotate(cv2.imread(os.path.join(datasets_dir, dataset, folder, image), cv2.IMREAD_GRAYSCALE),
                                 1)
            savepath = os.path.join(datasets_dir, dataset, folder, "rot180_" + image)
            print(savepath)
            cv2.imwrite(savepath, new_img)
            new_img = cv2.rotate(cv2.imread(os.path.join(datasets_dir, dataset, folder, image), cv2.IMREAD_GRAYSCALE),
                                 2)
            savepath = os.path.join(datasets_dir, dataset, folder, "rot270_" + image)
            print(savepath)
            cv2.imwrite(savepath, new_img)
            '''
        else:
            print("wrong if-folder check")
