import numpy as np
import os
import pandas as pd
import cv2
from scipy import ndimage
import skimage

SIMULATOR_HOME = "../data/"
DRIVING_LOG_FILE = "driving_log.csv"
DRIVING_LOG_FILE_PATH = os.path.join(SIMULATOR_HOME, DRIVING_LOG_FILE)

IMAGE_PATH = os.path.join(SIMULATOR_HOME, "IMG")

steering_offset = 0.2

driving_log = pd.read_csv(DRIVING_LOG_FILE_PATH)
driving_log.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

############################
# Functions for Loading data
############################

def load_data_from_frames():
    df_cn = driving_log.copy()[["center", "steering"]]
    df_cn.columns = ["image_path", "angle"]

    df_lf = driving_log.copy()[["left", "steering"]]
    df_lf.columns = ["image_path", "angle"]
    df_lf.angle += steering_offset

    df_rh = driving_log.copy()[["right", "steering"]]
    df_rh.columns = ["image_path", "angle"]
    df_rh.angle -= steering_offset

    df_all = pd.concat([df_cn, df_lf, df_rh]).reset_index(drop=True)
    return df_all

def load_training_validation_df(all_data):
    train_data = all_data.sample(frac=0.8, random_state=200123)
    validation_data = all_data.drop(train_data.index)
    return train_data, validation_data


def data_generator(df, batch_size=128, is_training=1):
    n_rows = df.shape[0]
    while True:
        # Shuffle the data frame rows after every complete cycle through the data
        df = df.sample(frac=1).reset_index(drop=True)

        for index in range(0, n_rows, batch_size):
            df_batch = df[index: index + batch_size]

            # Ignoring the last batch which is smaller than the requested batch size
            #if (df_batch.shape[0] == batch_size):
            X_batch = []
            y_batch = []
            for i , row in df_batch.iterrows():
                img = get_image(row) #row["image_path"].strip()
                angle = row["angle"]
                # Normal image
                X_batch.append(img)
                y_batch.append(angle)
                if is_training == 1:
                    # Normal with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(img)))
                    y_batch.append(angle)
                    # Flipped image
                    f_img = get_flipped_image(img)
                    X_batch.append(f_img)
                    y_batch.append(-angle)
                    # Flipped with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(f_img)))
                    y_batch.append(-angle)
                    # blurred image
                    b_img = get_blurred_image(img)
                    X_batch.append(b_img)
                    y_batch.append(angle)
                    # blurred with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(b_img)))
                    y_batch.append(angle)
                    # Flipped & Blurred image
                    f_b_img = get_blurred_image(get_flipped_image(img))
                    X_batch.append(f_b_img)
                    y_batch.append(-angle)
                    # Flipped & Blurred with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(f_b_img)))
                    y_batch.append(-angle)
                    # Speckled image
                    s_img = get_speckled_image(img)
                    X_batch.append(s_img)
                    y_batch.append(angle)
                    # Speckled with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(s_img)))
                    y_batch.append(angle)
                    # Flipped & Speckled image
                    f_s_img = get_blurred_image(get_flipped_image(img))
                    X_batch.append(f_s_img)
                    y_batch.append(-angle)
                    # Flipped & Speckled with random Translate and Rotate
                    X_batch.append(translateImage(rotateImage(f_s_img)))
                    y_batch.append(-angle)

            #X_batch = np.array([get_image(row) for i, row in df_batch.iterrows()])
            #y_batch = np.array([row['angle'] for i, row in df_batch.iterrows()])
            yield (np.array(X_batch), np.array(y_batch))



############################
# Functions for Loading Images
############################

def get_image(row):
    """
        For a given row of the df,
        get the Augmented image based on the operations specified
        in it's name
    """
    image_name = row["image_path"].strip()

    #ops = get_ops(image_name)

    #image_name = ops[0]

    #ops = ops[1:]

    image = cv2.imread(os.path.join(SIMULATOR_HOME, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    #for op in ops:
        #if op == "INV":
            #image = get_flipped_image(image)

        #elif op == "BLUR":
            #image = get_blurred_image(image)

        #elif op == "NOISE":
            #image = get_speckled_image(image)

    return image

def pre_process(image, top_prop=0.35, bottom_prop=0.1):
    """
        - Crop the top `top_prop` and the bottom `bottom_prop` of the image
        - Resize the image to half of it's original size
    """
    rows_to_crop_top = int(image.shape[0] * 0.4)
    rows_to_crop_bottom = int(image.shape[0] * 0.1)
    image = image[rows_to_crop_top:image.shape[0] - rows_to_crop_bottom, :]

    return cv2.resize(image, (0,0), fx=0.5, fy=0.5)

#############################
# Functions for Sampling Data
#############################

def sampling_data(df,num_bins = 23):
    angles = df["angle"]
    df_length = len(df.index)
    avg_samples_per_bin = df_length / num_bins
    hist, bins = np.histogram(angles, num_bins)
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1. / (hist[i] / target))
    remove_list = []
    for i in range(df_length):
        for j in range(num_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j + 1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    #df.drop(df.index[i], inplace=True)
                    remove_list.append(i)
    df.drop(df.index[[idx for idx in remove_list]], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    #image_paths = np.delete(image_paths, remove_list, axis=0)
    #angles = np.delete(angles, remove_list)


############################
# Functions for Augmentation
############################

def get_flipped_image(image):
    """
        returns image which is flipped about the vertical axis
    """
    return cv2.flip(image, 1)


def get_blurred_image(image):
    """
        Performs a gaussian blur on the image and returns it
    """
    return ndimage.gaussian_filter(image, sigma=1)


def get_speckled_image(image):
    """
        Adds random noise to an image
    """
    return skimage.img_as_ubyte(skimage.util.random_noise(image.astype(np.uint8), mode='gaussian'))


def translateImage(image):
    t_x = (np.random.randn(1)*.5)[0]
    t_y = (np.random.randn(1)*.5)[0]
    #print(t_x,t_y)
    rows,cols,_ = image.shape
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def rotateImage(image):
    theta = (np.random.randn(1)*5)[0]
    #print(theta)
    rows,cols,_ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst