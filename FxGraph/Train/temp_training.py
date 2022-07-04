from FxGraph.DataLoader.LitGraphData import LitGraphData
from FxGraph.Model.temp_torch_lighting import LitGraphModel, DeepGraph

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
import torch

torch.set_printoptions(precision=5)

logger = TensorBoardLogger("tb_logs", name="my_model")

config = {
    'batch_size': 32,
    'num_nodes': 6,
    'predict_task': 'prediction',
    'step_predict': 8,
    'step_share_history': 2,
    'split': (6, 2, 2),
    'window_temporal': 64,
    'hidden_channels': 256,
    'out_channels': 2 + 8,
    'num_layers': 8,
    'learning_rate': 1e-4,
}
data_module = LitGraphData(config)
model = DeepGraph(config)  #LitGraphModel(config)  #
dataloader = data_module.train_dataloader()

for i, data_batch in enumerate(dataloader):

    y = model(data_batch.x, data_batch.edge_index)
    print(y)
#     if i == 0:
#         break
#     t2 = data_batch.y
#     print(t2.reshape(16, 6))
#     t2 = t2.reshape(16, 6)
#     t2 = torch.tensor_split(t2, 6, dim=1)
#     loss_total = 0
#     for k in range(6):
#         loss_k = torch.nn.L1Loss()(y[k], t2[k])
#         loss_total += loss_k
#         print(loss_k)
#     print(loss_total)
#
    if i == 0:
        break
#
# if __name__ == '__main__':
# trainer = pl.Trainer(
#     #fast_dev_run=True,
#     gpus=1,
#     logger=logger,
#     max_epochs=300,
#     benchmark=True,
#     log_every_n_steps=1,
#     auto_lr_find=False,
# )
#     # lr_finder = Tuner(trainer).lr_find(model, train_dataloaders=data_module)
#     # model.hparams.lr = lr_finder.suggestion()
#     # print(f'Auto-find model LR: {model.hparams.lr}')
#
#     trainer.fit(model, datamodule=data_module)
