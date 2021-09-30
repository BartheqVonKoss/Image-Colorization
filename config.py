class Configuration:

    training_folder = 'data/training_imgs/'
    validation_folder = 'data/validation_imgs/'
    work_dir = 'work_dir/'

    HEIGHT = 256
    WIDTH = 256
    div = 0.5  # value by which the model is smaller than the one in the paper / 4 for same network
    bin_no = 50
    EPOCHS = 250
    batch_size = 8
    batch_normalization = True
    bias = True
    lr = 1e-3
    REG_COEF = 0
    just_testing = True
