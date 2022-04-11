import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.utils.data as Data

from models.models import CNN
import numpy as np
from post_processing.recorder import Recorder
from tqdm import trange

FEDPROX = False
DOWNLOAD_MNIST = False


class OemCloud(object):
    def __init__(self, name, epoch, mu):
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name
        # self.epoch = randrange(1, 20)
        self.epoch = epoch
        self.mu = mu
        self.batch_size = 50
        self.lr = 0.001
        self.num_workers = 6
        self.train_data = None
        self.test_data = None
        self.cnn = CNN()
        self.loss_rec = []
        self.n_data = 0
        gpu = 0
        self.device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_train_data(self, data):
        self.train_data = data

    def load_test_data(self, data):
        self.test_data = data

    def update(self, global_model_state):
        self.cnn = CNN()
        self.cnn.load_state_dict(global_model_state)

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
        train_loader = Data.DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)

        test_x = torch.FloatTensor(np.array(self.test_data['x']).reshape(len(self.test_data['x']), 28, 28))
        test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor).to(self.device)
        test_y = torch.FloatTensor(np.array(self.test_data['y'])).to(self.device)

        global_weight_collector = copy.deepcopy(list(self.cnn.to(self.device).parameters()))

        recorder = Recorder()
        t = trange(self.epoch, leave=True)
        for epoch in t:
            for b_x, b_y in train_loader:
                b_x = Variable(b_x).to(self.device)  # Tensor on GPU
                b_y = Variable(b_y.type(torch.LongTensor)).to(self.device)  # Tensor on GPU

                output = self.cnn(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                ret_loss = loss.clone()

                # for fedprox
                if 1:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(self.cnn.parameters()):
                        fed_prox_reg += (
                                    (self.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss += fed_prox_reg

                loss.backward()

                # for gradient clip

                # for group in optimizer.param_groups:
                #     for param in group['params']:
                #         if param.grad is not None:
                #             param.grad.data.clamp_(-0.5, 0.5)

                optimizer.step()
            # for testing
            test_output = self.cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)

            t.set_description('Client: %s' % self.name +
                              '| Train loss: %.4f' % loss.data.cpu().numpy() +
                              '| Test accuracy: %.4f' % accuracy.cpu().numpy())
            recorder.res['server']['iid_accuracy'].append(accuracy.cpu().numpy())
            recorder.res['server']['train_loss'].append(loss.data.cpu().numpy())
        self.cnn.cpu()
        ret_loss.data.cpu()

        return self.cnn.state_dict(), ret_loss.data.cpu().numpy(), self.n_data, recorder
