from images_generator import load_data_from_frames, load_training_validation_df, data_generator

if __name__ == "__main__":
    all_df = load_data_from_frames()
    print(all_df.shape)