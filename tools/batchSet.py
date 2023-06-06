import torch
import torch.utils.data as Data
import scipy.io as sio
import numpy as np

r""" Only for generating batches from .mat file
that lie in the MatLab smoke_data file. 

Images are in size of 48x48 and for whole-image-classification.
"""


def dataLoading(path_name, set_name):
    tmp_data = sio.loadmat(path_name)
    labels = tmp_data['label_'+set_name]
    labels[labels == -1] = 0
    labels = np.ravel(labels, 'F')  # squeeze along columns
    tmp_data = tmp_data['smp_'+set_name]
    data = tmp_data.transpose(2, 0, 1)
    data = np.expand_dims(data, axis=1)
    # if len(tmp_data.shape) == 3:
    #     channel_num = 1
    # else:
    #     channel_num = tmp_data.shape[2]
    # newsize = [len(labels), 1, tmp_data.shape[0], tmp_data.shape[1]]
    # data = np.zeros(newsize)
    # for i in range(len(labels)):
    #     data[i, 0, :, :] = tmp_data[:, :, i]
    # del tmp_data
    # data = np.expand_dims(data, len(data.shape))
    return data, labels


def batchSet(BATCH_SIZE, fea, labels, flag, num_workers):
    torch_dataset = Data.TensorDataset(torch.FloatTensor(fea), torch.LongTensor(labels))
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE, drop_last=True,
        shuffle=flag, num_workers=num_workers,
    )
    return loader


def main(BATCH_SIZE, path1, flag):
    torch.manual_seed(1)
    name = 'E://matlabCode//data//smoke_mat//'
    data, labels = dataLoading(name + path1, flag)
    # torch_dataset = Data.TensorDataset(data_tensor=tr_data, target_tensor=tr_labels)
    # tr_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
    # torch_dataset = Data.TensorDataset(data_tensor=ts_data, target_tensor=ts_labels)
    # ts_loader = Data.DataLoader(dataset=torch_dataset, batch_size=100,shuffle=False, num_workers=2)
    if flag == 'train':
        loader = batchSet(BATCH_SIZE, data, labels, True, 1)
    else:
        loader = batchSet(1, data, labels, False, 1)

    # scaler = sklearn.preprocessing.StandardScaler().fit(tr_data.dataset[0])
    # tr_data.dataset[0] = scaler.transform(tr_data.dataset[0])
    return loader

