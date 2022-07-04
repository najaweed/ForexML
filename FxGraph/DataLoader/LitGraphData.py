from abc import ABC
import itertools
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from FxGraph.DataLoader.Permutation import Permutation

import matplotlib.pyplot as plt


class LitGraphData(pl.LightningDataModule, ABC):
    def __init__(
            self,
            config: dict,
            path_csv: str = '/home/z/Desktop/imb_m1_v1000.csv',
    ):
        super().__init__()
        self.batch_size = config['batch_size']
        self.window_temporal = config['window_temporal']
        self.step_predict = config['step_predict']
        self.step_share = config['step_share_history']
        self.split = config['split']
        self.task = config['predict_task']
        self.num_classes = config['out_channels']
        self.df = self._read_csv(path_csv)
        self.symbols = self._get_symbols()
        self.train_loader, self.val_loader, self.test_loader = self._gen_data_loaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def _read_csv(path: str):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        # drop sum volume
        df = df.iloc[:, :-2]
        # drop
        return df

    def _get_symbols(self):
        symbols = []
        for sym in self.df.columns:
            if sym[:6] not in symbols:
                symbols.append(sym[:6])
        return symbols

    def _get_static_edges(self):
        edges = []
        for i, i_sym in enumerate(self.symbols):
            for j, j_sym in enumerate(self.symbols):
                if i != j:

                    cond1 = i_sym[:3] == j_sym[3:6] or i_sym[:3] == j_sym[:3]
                    cond2 = i_sym[3:6] == j_sym[3:6] or i_sym[3:6] == j_sym[:3]

                    if cond1 or cond2:
                        edges.append([i, j])
        edges = torch.LongTensor(edges).T

        return edges

    def _gen_graph_data(self, time_series: torch.Tensor):
        graph_data = []
        for i in range(time_series.shape[0] - (self.window_temporal + self.step_predict)):
            edge_index = self._get_static_edges()

            nodes_features = time_series[i: i + self.window_temporal, :].T
            y_target = time_series[
                       (i + self.window_temporal - self.step_share):i + self.window_temporal + self.step_predict, :].T
            if self.task == 'classification':
                # nodes_features = self._gen_node_feature(nodes_features)
                nodes_features = Permutation(nodes_features.numpy(), 4, 1).gen_prob()
                nodes_features = torch.tensor(nodes_features, dtype=torch.float, )
                y_target = self._gen_target_label(y_target)
            elif self.task == 'prediction':
                y_target = self._normalize(y_target)

            # nodes_features = self._normalize(nodes_features)
            g_data = Data(x=nodes_features, edge_index=edge_index, y=y_target)
            graph_data.append(g_data)
        return graph_data

    def _gen_data_loaders(self):
        split_index = [int(self.df.shape[0] * self.split[i] / 10) for i in range(3)]
        s_index = 0
        time_series = torch.FloatTensor(self.df[[f'{sym}.si,close' for sym in self.symbols]].values)
        data_loaders = []
        for i in range(3):
            e_index = split_index[i] + s_index
            dataset = self._gen_graph_data(time_series[s_index:e_index, ...])
            data_loaders.append(DataLoader(dataset=dataset,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           drop_last=True,
                                           pin_memory=True,
                                           shuffle=False,
                                           ))
            s_index = e_index
        return data_loaders

    @staticmethod
    def _normalize(time_series):
        x_min = torch.amin(time_series, dim=1, keepdim=True)
        x_max = torch.amax(time_series, dim=1, keepdim=True)
        x = (time_series - x_min) / (x_max - x_min)
        return x

    def _gen_target_label(self, target,):
        all_perms = list(itertools.permutations(np.arange(0, self.step_predict), self.step_predict))
        target = target.numpy()
        target = np.argpartition(target, kth=self.step_predict - 1, axis=1)
        target_label = torch.zeros(target.shape[0], dtype=torch.long)
        for i in range(target.shape[0]):
            # print('i',i)
            # for i, x_i in range(target):
            for k, perm_k in enumerate(all_perms):
                # print('k',k)
                # print(tuple(target[i, ...]))
                # print(perm_k)
                if tuple(target[i, ...]) == perm_k:
                    # print(target[i, ...], k)
                    target_label[i] = k

        return target_label


#
# torch.set_printoptions(precision=5)
# config = {
#     'batch_size': 64,
#     'num_nodes': 6,
#     'predict_task': 'classification',
#     'step_predict': 3,
#     'step_share_history': 0,
#     'split': (6, 2, 2),
#     'window_temporal': 1,
#     'hidden_channels': 256,
#     'out_channels': 3,
#     'num_layers': 8,
#     'learning_rate': 1e-3,
# }
#
# path = '/home/z/Desktop/imb_m1_v1000.csv'
# data = LitGraphData(config, path)
#
# dataloader = data.train_dataloader()
# rand_loss = 0
# # from Permutation import Permutation
# # import numpy as np
#
# for j, data_batch2 in enumerate(dataloader):
#     t2 = data_batch2.x
#     # print(t2, t2.shape)
#     print(t2)
#     # print(np.argpartition(t2.numpy(), kth=8, axis=1))
#     print(t2.shape)
#     #
#     # t2 = torch.unsqueeze(t2, 0)
#     t2 = t2.reshape(config['batch_size'], config['num_nodes'], ).T
#
#     print(t2)
#     print(Permutation(t2, 4, 1).gen_prob())
#     probs = Permutation(t2, 4, 1).gen_prob()
#     print(probs.shape)
#     # print(np.sum(np.where(probs[1] != 0, probs[0] * np.log(probs[0] / probs[1]), 0)))
#     # import time
#     #
#     # s_time = time.time()
#     # for i in range(t2.shape[0]):
#     #     print(t2[i, ...])
#     #     perm = Permutation(np.squeeze(t2[i, ...]), 4, 1)
#     #     print(time.time() - s_time)
#     #     print(perm.embed)
#     #     print(np.argpartition(t2[i,...].numpy(),kth=3,axis=1))
#     # print(t2)
#     # print(torch.squeeze(t2[0, 0, :]))
#     import matplotlib.pyplot as plt
#
#     # target = torch.squeeze(t2[0, 0, :])
#     # rand = torch.rand_like(target)  # target + 0.6*torch.randn_like(target)
#     # y_hat = (rand - torch.min(rand)) / (torch.max(rand) - torch.min(rand))
#     # rand_loss += torch.nn.L1Loss()(target, y_hat)
#     # print(torch.nn.L1Loss()(target,y_hat))
#     # plt.plot(target)
#     # plt.plot(y_hat)
#     print(j)
#     # plt.show()
#     # t2 = torch.tensor_split(t2, config['num_nodes'], dim=1)
#     # print(t2[0].shape)
#     if j == 10:
#         print(rand_loss / 1)
#         break
