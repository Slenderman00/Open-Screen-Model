import papermill as pm

# Define a list of models with varying parameters
models = [
    {
        'name': 'openscreen-512-2-6-128-0.000125',
        'image_resolution': 512,
        'batch_size': 2,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-512-4-6-128-0.000125',
        'image_resolution': 512,
        'batch_size': 4,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-512-2-8-128-0.000125',
        'image_resolution': 512,
        'batch_size': 2,
        'depth': 8,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-512-2-6-256-0.000125',
        'image_resolution': 512,
        'batch_size': 2,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },

    # 256x256
    {
        'name': 'openscreen-256-2-6-128-0.000125',
        'image_resolution': 256,
        'batch_size': 2,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-256-4-6-128-0.000125',
        'image_resolution': 256,
        'batch_size': 4,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-256-2-8-128-0.000125',
        'image_resolution': 256,
        'batch_size': 2,
        'depth': 8,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    },
    {
        'name': 'openscreen-256-2-6-256-0.000125',
        'image_resolution': 256,
        'batch_size': 2,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 100
    }
]

for model in models:
    output_notebook = f"{model['name']}_output.ipynb"
    pm.execute_notebook(
        'model.ipynb',  # Input notebook
        output_notebook,  # Output notebook
        parameters=model,  # Parameters to be injected
    )