from abc import ABC

import torch
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl

from torch_geometric.nn import DeepGCNLayer, EdgeConv, GATConv
from torch_geometric_temporal.nn import ChebConvAttention, DCRNN, TGCN
import torch_geometric
from FxGraph.DataLoader.Permutation import Permutation


class DeepGraph(nn.Module):
    def __init__(self,
                 config: dict,
                 ):
        super().__init__()
        self.in_channels = config['in_channels']  # config['window_temporal']
        self.hidden_channels = config['hidden_channels']
        self.num_layers = config['num_layers']
        self.num_nodes = config['num_nodes']
        self.batch_size = config['batch_size']
        self.out_channels = config['out_channels']
        self.task = config['predict_task']
        # Pretraining/Preprocess layers
        # self.linear1 = nn.Linear(2 * self.in_channels, self.hidden_channels)
        # self.linear2 = nn.Linear(2 * self.hidden_channels, self.hidden_channels)

        self.gnn_layer = GATConv(self.in_channels, self.hidden_channels, )

        # Residual block
        self.residual_block = nn.ModuleList()
        if self.num_layers > 0:
            for i in range(1, self.num_layers + 1):
                conv = GATConv(self.hidden_channels, self.hidden_channels, )
                norm = nn.LayerNorm(self.hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)
                layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.4, ckpt_grad=i % 3)
                self.residual_block.append(layer)

        # Predictor/Classifier Layers
        self.predictor_block = nn.ModuleList()
        for _ in range(self.num_nodes):
            out_layer = torch.nn.Sequential(
                nn.Linear(self.hidden_channels, int(self.hidden_channels / 2), bias=False),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(int(self.hidden_channels / 2), self.out_channels, bias=False),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(int(self.hidden_channels / 4), self.out_channels, bias=False),
            )

            self.predictor_block.append(out_layer)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, edge_index):

        x = self.gnn_layer(x, edge_index).relu()

        # residual blocks
        if self.num_layers > 0:
            for layer in self.residual_block.children():
                x = layer(x, edge_index, )
            x = self.residual_block[0].act(self.residual_block[0].norm(x))

        # predictor/classifier
        x = x.reshape(self.num_nodes, self.batch_size, self.hidden_channels)
        x = torch.tensor_split(x, self.num_nodes, dim=0)
        list_x = []
        for i, x_i in enumerate(x):
            x_i = torch.squeeze(x_i)
            x_i = self.predictor_block[i](x_i)
            x_i = self.soft(x_i)
            list_x.append(x_i)

        return list_x

    @staticmethod
    def _normalize(time_series):
        x_min = torch.amin(time_series, dim=1, keepdim=True)
        x_max = torch.amax(time_series, dim=1, keepdim=True)
        x = (time_series - x_min) / (x_max - x_min)
        return x


class LitGraphModel(pl.LightningModule, ABC):

    def __init__(self,
                 config: dict,
                 ):
        super().__init__()

        # configuration
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_nodes = config['num_nodes']
        self.out_channels = config['out_channels']
        # model initialization
        self.nn_model: nn.Module = DeepGraph(config)
        self.l1_loss = torch.nn.L1Loss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch.x
        # x = x.reshape(self.batch_size, self.num_nodes, ).T
        # x = Permutation(x.numpy(), 4, 1).gen_prob()
        # x = torch.tensor(x,dtype=torch.float,)
        edge_index = train_batch.edge_index

        y = train_batch.y  # .view(-1, 1)
        # process model
        x = self.nn_model(x, edge_index)
        # criterion
        loss = self._calculate_loss(x, y, mode='train')  # F.l1_loss(x, y)
        # logger
        metrics = {'total_train_loss': loss, }
        self.log_dict(metrics)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.x
        # x = x.reshape(self.batch_size, self.num_nodes, ).T
        # x = Permutation(x.numpy(), 4, 1).gen_prob()
        # x = torch.tensor(x,dtype=torch.float,)
        edge_index = val_batch.edge_index

        y = val_batch.y  # .view(-1, 1)
        # process model
        x = self.nn_model(x, edge_index)
        # criterion
        loss = self._calculate_loss(x, y, mode='val')  # F.l1_loss(x, y)

        # logger
        metrics = {'total_val_loss': loss, }

        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["total_val_loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", float(avg_loss.detach().cpu()))

    def _calculate_loss(self, y_model, y_target, mode='train'):
        # y_target = y_target.reshape(self.batch_size, self.num_nodes)
        # y_target = torch.tensor_split(y_target, self.num_nodes, dim=1)
        # y_target = torch.unsqueeze(y_target, dim=0)
        y_target = y_target.reshape(self.batch_size, self.num_nodes)

        loss_total = 0
        loss_k = 0
        for k in range(self.num_nodes):
            # print(y_model[k], y_target[:, k])
            if mode == 'val':
                loss_k = self.ce_loss(y_model[k], torch.squeeze(y_target[:, k]))
            elif mode == 'train':
                loss_k = self.ce_loss(y_model[k], torch.squeeze(y_target[:, k]))

            # loss_k = self.l1_loss(y_model[k], y_target[k])
            self.log_dict({f'{mode}/mode_{k}_loss': loss_k, })
            loss_total += loss_k

        return loss_total
