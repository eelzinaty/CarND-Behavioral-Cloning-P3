import os
import argparse
from images_generator import load_data_from_frames, load_training_validation_df, data_generator, sampling_data
import numpy as np

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
# Fix error with TF and Keras
import tensorflow as tf

tf.python.control_flow_ops = tf

SAVE_HOME = "./outputs/steering_model/"
MODEL_HOME = os.path.join(SAVE_HOME, "model.h5")
WEIGHT_HOME = os.path.join(SAVE_HOME, "weight.json")

# model developed by comma.ai
def get_comma_ai_model(shape):
    model = Sequential()
    #model.add(Lambda(lambda x: x / 127.5 - 1.,input_shape=shape))
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return model

# model developed by Nvidia
def get_nvidia_model(shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    # parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    # parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    # parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
    parser.add_argument('--visualize', type=int, default=0, help='Only visualize data.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()

    all_df = load_data_from_frames()

    if args.visualize == 1:
        from visualize import visualize_data_histogram, visualize_p_data_histogram

        #visualize_data_histogram(np.array([row['angle'] for i, row in all_df.iterrows()]))
        #visualize_p_data_histogram(all_df)
        all_df = sampling_data(all_df)
        visualize_p_data_histogram(all_df)
        #sampling_data(all_df)

    else:
        all_df = sampling_data(all_df)
        training_df, validation_df = load_training_validation_df(all_df)
        n = training_df.shape[0]
        batch_size = 64
        samples_per_epoch = int(n / batch_size)

        # Create training and validation generators
        train_gen = data_generator(training_df, batch_size=batch_size)
        validation_gen = data_generator(validation_df, batch_size=batch_size)

        #X_batch, y_batch = next(train_gen)
        input_shape = (160, 320, 3)
        model = get_comma_ai_model(input_shape)

        if not os.path.exists(SAVE_HOME):
            os.makedirs(SAVE_HOME)

        # Create checkpoint at which model weights are to be saved
        checkpoint = ModelCheckpoint(MODEL_HOME, monitor='val_loss', verbose=0,
                                     save_best_only=True, save_weights_only=False, mode='auto')

        # Train the model
        model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch * batch_size, nb_epoch=10, callbacks=[checkpoint],
                            validation_data=validation_gen, nb_val_samples=validation_df.shape[0])

        # Save the model architecture
        with open(WEIGHT_HOME, "w") as file:
            file.write(model.to_json())

        print("Model has been trained and saved.")



        #model.fit_generator(
            #gen(20, args.host, port=args.port),
            #samples_per_epoch=10000,
            #nb_epoch=args.epoch,
            #validation_data=gen(20, args.host, port=args.val_port),
            #nb_val_samples=1000
        #)



