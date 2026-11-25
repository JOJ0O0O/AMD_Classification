n_folds = 2 #Number of folds

#Choose which size the images get resized to
image_size = (64, 64)

#Parameter Grid configuration
param_grid = {
        'module__channels': [(32, 64),(32, 64, 128), (64, 128, 256)],
        'module__kernel_size': [3],
        'lr': [0.001],
        'max_epochs': [5],
        'optimizer__weight_decay': [0.0],
        'batch_size': [64],
        'module__hidden_features' : [128, 256]
    }