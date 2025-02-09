import papermill as pm
from concurrent.futures import ProcessPoolExecutor
import logging

# Define a smaller set of hyperparameters to experiment with
models = [
    # Baseline model
    {
        'name': 'baseline-256-4-4-128',
        'image_resolution': 256,
        'batch_size': 4,
        'depth': 4,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # Higher resolution and depth
    {
        'name': 'highres-512-4-6-128',
        'image_resolution': 512,
        'batch_size': 4,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # Larger batch size
    {
        'name': 'largebatch-256-8-6-128',
        'image_resolution': 256,
        'batch_size': 8,
        'depth': 6,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # More initial features
    {
        'name': 'morefeatures-256-4-6-256',
        'image_resolution': 256,
        'batch_size': 4,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # Deeper model
    {
        'name': 'deeper-256-4-8-128',
        'image_resolution': 256,
        'batch_size': 4,
        'depth': 8,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # Combination of high resolution and more features
    {
        'name': 'combo-512-4-6-256',
        'image_resolution': 512,
        'batch_size': 4,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # High resolution, large batch size
    {
        'name': 'highres-largebatch-512-8-4-128',
        'image_resolution': 512,
        'batch_size': 8,
        'depth': 4,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # High resolution, deeper model
    {
        'name': 'highres-deeper-512-4-8-128',
        'image_resolution': 512,
        'batch_size': 4,
        'depth': 8,
        'init_features': 128,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # High resolution, more features
    {
        'name': 'highres-morefeatures-512-4-6-256',
        'image_resolution': 512,
        'batch_size': 4,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    },
    # High resolution, large batch size, more features
    {
        'name': 'highres-largebatch-morefeatures-512-8-6-256',
        'image_resolution': 512,
        'batch_size': 8,
        'depth': 6,
        'init_features': 256,
        'learning_rate': 0.000125,
        'debug': False,
        'debug_print_model': True,
        'epochs': 25
    }
]

num_workers = 4


logging.basicConfig(level=logging.INFO)


def train_model(model):
    logging.info(f"Starting training for model: {model['name']}")
    output_notebook = f"{model['name']}_output.ipynb"
    pm.execute_notebook(
        'model.ipynb',
        output_notebook,
        parameters=model,
    )
    logging.info(f"Completed training for model: {model['name']}")


with ProcessPoolExecutor(max_workers=num_workers) as executor:
    executor.map(train_model, models)