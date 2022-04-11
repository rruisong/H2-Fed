import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.utils.data as Data
from models.models import CNN
import numpy as np

FEDPROX = False
DOWNLOAD_MNIST = False


class FedClient(object):
    def __init__(self, name, epoch, mu):
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name
        self.epoch = epoch
        self.mu = mu
        self.batch_size = 50
        self.lr = 0.001
        self.num_workers = 2
        self.train_data = {'x': [], 'y': []}
        self.test_data = None
        self.cnn = CNN()
        model_parameters = filter(lambda p: p.requires_grad, self.cnn.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        self.rsu_model = CNN()
        self.cloud_model = CNN()
        self.loss_rec = []
        self.n_data = 0
        self.rsu_id = None
        gpu = 0
        self.device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_train_data(self, data):

        self.train_data['x'] = data['x']
        self.train_data['y'] = data['y']

    def load_test_data(self, data):
        self.test_data = data

    def local_update(self, rsu_model_state, cloud_model_state=None):
        self.cnn = CNN()
        self.cnn.load_state_dict(rsu_model_state)
        self.rsu_model.load_state_dict(rsu_model_state)
        if cloud_model_state is not None:
            self.cloud_model.load_state_dict(cloud_model_state)

    def train(self):

        self.cnn.to(self.device)
        # optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.cnn.parameters(), lr=self.lr)

        loss_func = nn.CrossEntropyLoss()

        data_x = torch.FloatTensor(np.array(self.train_data['x']).reshape(len(self.train_data['x']), 28, 28))
        data_x = torch.unsqueeze(data_x, dim=1).type(torch.FloatTensor)
        data_y = torch.FloatTensor(np.array(self.train_data['y']))
        self.n_data = len(self.train_data['y'])
        data_set = Data.TensorDataset(data_x, data_y)
        train_loader = Data.DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_x = torch.FloatTensor(np.array(self.test_data['x']).reshape(len(self.test_data['x']), 28, 28))
        test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor).to(self.device)
        test_y = torch.FloatTensor(np.array(self.test_data['y'])).to(self.device)

        rsu_weights = copy.deepcopy(list(self.rsu_model.to(self.device).parameters()))
        global_weights = copy.deepcopy(list(self.cloud_model.to(self.device).parameters()))

        epoch_loss_collector = []
        for epoch in range(self.epoch):
            epoch_loss_collector = []
            for b_x, b_y in train_loader:
                b_x = Variable(b_x).to(self.device)  # Tensor on GPU
                b_y = Variable(b_y.type(torch.LongTensor)).to(self.device)  # Tensor on GPU

                output = self.cnn(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()

                prox_term = 0.0
                for p_i, param in enumerate(self.cnn.parameters()):
                    prox_term += (((self.mu['mu1'] * self.param_len / 2) * torch.norm((param - rsu_weights[p_i])) ** 2)
                                     + ((self.mu['mu2'] * self.param_len / 2) * torch.norm((param - global_weights[p_i])) ** 2))
                loss += prox_term
                loss.backward()
                optimizer.step()
                epoch_loss_collector.append(loss.item())
        # for testing
        # test_output = self.cnn(test_x)
        # pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # accuracy = sum(pred_y == test_y) / test_y.size(0)

        # print('Client: %s', self.name,
        #       '| Train loss: %.4f' % loss.data.cpu().numpy(),
        #       '| Test accuracy: %.4f' % accuracy.cpu().numpy())

        self.cnn.cpu()
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        return self.cnn.state_dict(), epoch_loss, self.n_data
