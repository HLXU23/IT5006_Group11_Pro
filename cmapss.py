import torch
import tsfresh
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset

class cmapss(Dataset):
    def __init__(self, mode='train', feature_extract_mode='default', dataset=None, rul_result=None, window_size=30, max_rul=125):
        self.data_df = pd.read_csv(dataset)
        self.data = self.data_df.values.astype(np.float32)
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
        print(f'Run {feature_extract_mode} feature extraction for {self.mode} dataset')

        # Feature extraction
        features = []
        if feature_extract_mode == 'tsfresh':
            features = self.tsfresh_feature_extract()
        else:
            dataset = self.x
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

    def tsfresh_feature_extract(self):
        feature_file_path = './feature_statistics.txt'
        features = []

        with open(feature_file_path, 'r') as file:
            feature_lines = file.readlines()

        feature_config = []
        for line in feature_lines:
            parts = line.strip().split('__')
            if len(parts) == 2:
                column_name, feature_part = parts[0], parts[1]
                column_name = column_name.replace(' ', '_')
                feature_name = feature_part.split(':')[0].strip()
                feature_config.append((column_name, feature_name))

        dataset = self.x
        x_cols = self.data_df.columns[2:]

        feature_extractors = {
            'minimum': extract_minimum,
            'maximum': extract_maximum,
            'absolute_maximum': extract_absolute_maximum,
            'median': extract_median,
            'sum_values': extract_sum_values,
            'standard_deviation': extract_standard_deviation,
            'variance': extract_variance,
            'root_mean_square': extract_root_mean_square
        }

        for unit_id in tqdm(range(dataset.shape[0]), desc="Processing units", unit="unit"):
            data = dataset[unit_id]

            x_df = pd.DataFrame(data)
            x_df.columns = x_cols

            x_df.columns = [col.replace(' ', '_') for col in x_df.columns]

            data_features = []
            for col_name, feature_type in feature_config:
                if col_name in x_df.columns:
                    if feature_type in feature_extractors:
                        feature_func = feature_extractors[feature_type]
                        extracted_feature = feature_func(x_df[col_name].values)

                        data_features.append(extracted_feature)
                    else:
                        data_features.append(np.nan)

            features.append(data_features)

        return features

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


def extract_minimum(data):
    return np.min(data)

def extract_maximum(data):
    return np.max(data)

def extract_absolute_maximum(data):
    return np.max(np.abs(data))

def extract_median(data):
    return np.median(data)

def extract_sum_values(data):
    return np.sum(data)

def extract_standard_deviation(data):
    return np.std(data)

def extract_variance(data):
    return np.var(data)

def extract_root_mean_square(data):
    return np.sqrt(np.mean(np.square(data)))




