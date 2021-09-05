sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [2, 5, 10]
        },
        'batch_size': {
            'values': [256, 128, 64, 32]
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'fc_layer_size':{
            'values':[128,256,512]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
    }
}