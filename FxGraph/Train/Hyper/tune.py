from FxGraph.DataLoader.LitGraphData import LitGraphData
from FxGraph.Model.temp_torch_lighting import LitGraphModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

logger = TensorBoardLogger("/home/z/PycharmProjects/ForexGraph/FxGraph/Train/tb_logs", name="my_model")

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

import json


def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet


def lit_model_train(config, num_epochs=10, ):
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=1,
        logger=logger,
        benchmark=True,
        log_every_n_steps=1,
        enable_progress_bar=False,
        callbacks=[TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end")]
    )
    data_module = LitGraphData(config)
    model = LitGraphModel(config)
    lr_finder = Tuner(trainer).lr_find(model, train_dataloaders=data_module)
    model.hparams.lr = lr_finder.suggestion()
    print(f'Auto-find model LR: {model.hparams.lr}')

    trainer.fit(model, datamodule=data_module)


def lit_tune(num_samples=50, num_epochs=1, ):

    config = {
        'batch_size': tune.choice([2, 4, 8]),
        'num_nodes': 6,
        'predict_task': 'classification',
        'window_temporal': tune.choice([64, 32, 8, 16]),
        'step_predict': 8,
        'split': (6, 2, 2),
        'hidden_channels': tune.choice([64, 128, 256]),
        'nun_classes': 3,
        'num_layers': tune.choice([1, 2, 4]),
        'learning_rate': 1e-3,
    }

    lit_tune_able = tune.with_parameters(lit_model_train, num_epochs=num_epochs)

    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(
        lit_tune_able,
        resources_per_trial={
            "cpu": 6,
            "gpu": 0.3
        },
        metric="loss",
        mode="min",
        # verbose=0,
        config=config,
        scheduler=scheduler,
        num_samples=num_samples,
        name="tune_mnist")

    print("Best hyper parameters found were: ", analysis.best_config)
    filename = 'best_hparams'
    with open(filename, 'w') as f:
        f.write(json.dumps(analysis.best_config))


lit_tune(num_samples=100, num_epochs=10)
