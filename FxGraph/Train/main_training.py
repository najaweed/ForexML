from FxGraph.DataLoader.LitGraphData import LitGraphData
from FxGraph.Model.temp_torch_lighting import LitGraphModel
import json

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(

    gpus=0,
    logger=logger,
    max_epochs=300,
    benchmark=True,
    log_every_n_steps=1,
    auto_lr_find=False,
)


def get_hyper_param():
    with open('Hyper/best_hparams') as f:
        config = None  # json.loads(f.read())
        if config is None:
            print('set default hyper params')

            config = {
                'batch_size': 32,
                'num_nodes': 6,
                'predict_task': 'classification',
                'step_predict': 3,
                'step_share_history': 0,
                'split': (6, 2, 2),
                'window_temporal': 128,
                'in_channels': 2 * 3 * 4,
                'hidden_channels': 256,
                'out_channels': 2 * 3,
                'num_layers': 8,
                'learning_rate': 5e-3,
            }
            print(config)

        else:
            print('set tuned hyper params')
            print(config)
    return config


if __name__ == '__main__':
    config = get_hyper_param()
    data_module = LitGraphData(config)
    model = LitGraphModel(config)
    lr_finder = Tuner(trainer).lr_find(model, train_dataloaders=data_module)
    model.hparams.lr = lr_finder.suggestion()
    print(f'Auto-find model LR: {model.hparams.lr}')

    trainer.fit(model, datamodule=data_module)
