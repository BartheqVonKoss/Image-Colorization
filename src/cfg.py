class Configuration:

    training_folder = '/home/bartheq/Documents/playground/image_colorization/data/training_images/'
    validation_folder = '/home/bartheq/Documents/playground/image_colorization/data/validation_images/'
    work_dir = '/home/bartheq/Documents/playground/image_colorization/work_dir/'


    HEIGHT = 256
    WIDTH = 256
    div = 1  # value by which the model is smaller than the one in the paper / 4 for same network
    batch_normalization = True
    bias = True

