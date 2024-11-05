import torch
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset

class cmapss(Dataset):
    def __init__(self, mode='train', feature_extract_mode='default', dataset=None, rul_result=None, window_size=30, max_rul=125):
        self.data = np.loadtxt(fname=dataset, dtype=np.float32)
        # Delete sensor 1, 5, 6, 10, 16, 18, 19
        self.data = np.delete(self.data, [5, 9, 10, 14, 20, 22, 23], axis=1)
        self.sample_num = int(self.data[-1][0])
        self.length = []
        self.mode = mode
        self.feature_extract_mode = feature_extract_mode
        self.window_size = window_size
        self.max_rul=max_rul
        self.features = []

        if self.mode == 'test' and rul_result is not None:
            self.rul_result = np.loadtxt(fname=rul_result, dtype=np.float32)
        if self.mode == 'test' and rul_result is None:
            raise ValueError('You did not specify the rul_result file path of the testset, '
                             'please check if the parameters you passed in are correct.')
        if self.mode != 'test' and self.mode != 'train':
            raise ValueError('You chose an undefined mode, '
                             'please check if the parameters you passed in are correct.')
        if self.mode == 'train' and rul_result is not None:
            warnings.warn('This rul_result file will only be used in the test set, '
                          'and the current mode you selected is training, so the file will be ignored.')

        self.x = []
        self.y = []

        if self.mode == 'train':
            for i in range(1, self.sample_num + 1):
                ind = np.where(self.data[:, 0] == i)
                # transfer tuple to ndarray
                ind = ind[0]
                # single engine data
                data_temp = self.data[ind, :]
                for j in range(len(data_temp) - self.window_size + 1):
                    self.x.append(data_temp[j: j+self.window_size, 2:])
                    rul = len(data_temp) - self.window_size - j
                    if rul > self.max_rul:
                        rul = self.max_rul
                    self.y.append(rul)

        if self.mode == 'test':
            for i in range(1, self.sample_num + 1):
                ind = np.where(self.data[:, 0] == i)[0]
                data_temp = self.data[ind, :]

                if len(data_temp) < self.window_size:
                    data = np.zeros((self.window_size, data_temp.shape[1]))
                    for j in range(data.shape[1]):
                        x_old = np.linspace(0, len(data_temp)-1, len(data_temp), dtype=np.float64)
                        params = np.polyfit(x_old, data_temp[:, j].flatten(), deg=1)
                        k = params[0]
                        b = params[1]
                        x_new = np.linspace(0, self.window_size-1, self.window_size, dtype=np.float64)
                        data[:, j] = (x_new * len(data_temp) / self.window_size * k + b)
                    self.x.append(data[-self.window_size:, 2:])

                else:
                    self.x.append(data_temp[-self.window_size:, 2:])

                rul = self.rul_result[i - 1]
                if rul > self.max_rul:
                    rul = self.max_rul
                self.y.append(rul)
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)/self.max_rul
        

    def feature_extract(self):

        feature_extract_mode = self.feature_extract_mode
        dataset = self.x
        print(f'Run {feature_extract_mode} feature extraction for {self.mode} dataset')

        # Feature extraction
        features = []
        if feature_extract_mode == 'tsfresh':
            features = []
        else:
            for data in tqdm(dataset, desc="Processing data", unit="data"):
                x = np.array(range(data.shape[0]))
                features_for_data = []
                for i in range(data.shape[1]):
                    mean_val = np.mean(data[:, i])
                    slope = np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0]
                    features_for_data.append(mean_val)
                    features_for_data.append(slope)
                features.append(features_for_data)

        # Normalization
        features = np.array(features)
        feature_mean = np.mean(features, axis=0)
        feature_sigma = np.std(features, axis=0)
        eps = 1e-10
        normalized_features = (features - feature_mean) / (feature_sigma + eps)
        self.features = normalized_features
        return normalized_features

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        if self.features is not None and len(self.features) > 0:
            handcrafted_features = torch.from_numpy(self.features[index]).to(torch.float32)
        else:
            handcrafted_features = torch.empty(0, dtype=torch.float32)
        return x_tensor, handcrafted_features, y_tensor



