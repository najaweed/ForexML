from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl


class GraphDataset:
    def __init__(self,
                 c_symbols: list,
                 dataset,
                 c_window_temporal,
                 c_step_prediction,
                 c_alpha_select_position=1.01,
                 ):
        self.symbols = c_symbols
        self.static_edges = self._get_static_edges()
        self.window_temporal = c_window_temporal
        self.step_prediction = c_step_prediction
        self.alpha_position = c_alpha_select_position
        self._dataset = dataset

    # TODO separation of obs-target at end of day - to avoid prediction based on last day
    def _get_static_edges(self):
        edges = []
        for i, i_sym in enumerate(self.symbols):
            for j, j_sym in enumerate(self.symbols):
                if i > j:

                    cond1 = i_sym[:3] == j_sym[3:6] or i_sym[:3] == j_sym[:3]
                    cond2 = i_sym[3:6] == j_sym[3:6] or i_sym[3:6] == j_sym[:3]

                    if cond1 or cond2:
                        edges.append([i, j])
        edges = np.array(edges)

        return np.array(edges).T

    def data_loader(self, batch_size=1, shuffle=False):
        all_data = []
        for i in range(self._dataset.shape[0] - (self.window_temporal + self.step_prediction)):
            edge_index = torch.tensor(self.static_edges, dtype=torch.long)

            nodes_features = torch.tensor(self._dataset[i: i + self.window_temporal, :].T, dtype=torch.float)
            y_target = torch.tensor(self._dataset[i + self.window_temporal, :].T, dtype=torch.float)

            g_data = Data(x=nodes_features, edge_index=edge_index, y=y_target)
            all_data.append(g_data)

        return DataLoader(dataset=all_data,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=True,
                          pin_memory=True,
                          num_workers=15
                          )


def gen_graph_data_loader(
        df_rates,
        symbols,
        batch_size=15,
        window_temporal=180,
        step_predict=5,
        split=(6, 2, 2),
        step_share=0,
        shuffle=False, ):
    data_loaders = []
    split_index = [int(df_rates.shape[0] * split[i] / 10) for i in range(3)]
    s_index = 0
    data_price = df_rates[[f'{sym},close' for sym in symbols]].to_numpy()

    for i in range(3):
        e_index = split_index[i] + s_index
        fin_features = data_price[s_index:e_index, ]
        my_dataset = GraphDataset(symbols, fin_features, window_temporal, step_predict)

        data_loader = my_dataset.data_loader(batch_size=batch_size)
        data_loaders.append(data_loader)
        s_index = e_index
    return data_loaders


# currencies = ['EUR', 'USD', 'GBP', 'CHF']
# symbols = ['EURUSD.si', 'EURGBP.si', 'GBPUSD.si', 'EURCHF.si', 'USDCHF.si', 'GBPCHF.si']
#
#
# df = pd.read_csv('F:\\imb_m1_v1000.csv', index_col=0)
# df.index = pd.to_datetime(df.index)
# df.drop(columns='sum_volume', inplace=True)
#
# BATCH_SIZE = 1
# WINDOW_TEMPORAL = 16
# data_loader, val_loader, test_loader = gen_graph_data_loader(df,  symbols,
#                                                              batch_size=BATCH_SIZE,
#                                                              window_temporal=WINDOW_TEMPORAL,
#                                                              step_predict=1,
#                                                              )
# data= next(iter(data_loader))
# print(data.x)
# print(data.y)


class LitGraphData(pl.LightningDataModule, ABC):
    def __init__(
            self,
            config: dict,
    ):
        super().__init__()
        self.batch_size = config['batch_size']
        self.window_temporal = config['window_temporal']
        self.step_predict = config['step_predict']

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass



