import numpy as np
from models.models import CNN
import torch


class TrafficCloud(object):
    def __init__(self, global_rsu_list):

        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_rsu_ids = []

        self.global_client_list = global_rsu_list

        self.test_data = {}

        self.model = CNN()
        self.global_model = CNN()

        self.round = 0
        self.n_data = 0

    def load_test_data(self, data):
        self.test_data = data

    def test(self):

        test_x = torch.FloatTensor(np.array(self.test_data['x']).reshape(len(self.test_data['x']), 28, 28))
        test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
        test_y = torch.FloatTensor(np.array(self.test_data['y']))

        test_output = self.model(test_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == test_y) / test_y.size(0)

        return accuracy.numpy()

    def select_clients(self, connection_ratio):
        # select a fraction of clients
        self.selected_rsu_ids = []
        for client_id in self.global_client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_rsu_ids.append(client_id)
        # print(self.selected_rsu_ids)

    def aggregate(self):
        client_num = len(self.selected_rsu_ids)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0
        cnn = CNN()
        model_state = cnn.state_dict()
        avg_loss = 0
        # print('number of selected clients in Cloud: ' + str(client_num))
        for i, rsu_id in enumerate(self.selected_rsu_ids):
            if rsu_id not in self.client_state:
                continue
            for key in self.client_state[rsu_id]:
                if i == 0:
                    model_state[key] = self.client_state[rsu_id][key] * self.client_n_data[rsu_id] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[rsu_id][key] * self.client_n_data[
                        rsu_id] / self.n_data

            avg_loss = avg_loss + self.client_loss[rsu_id] * self.client_n_data[rsu_id] / self.n_data

        self.model.load_state_dict(model_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data

    def receive(self, rsu_id, state_dict, n_data, loss):
        self.n_data = self.n_data + n_data
        self.client_state[rsu_id] = {}
        self.client_n_data[rsu_id] = {}
        self.client_state[rsu_id].update(state_dict)
        self.client_n_data[rsu_id] = n_data
        self.client_loss[rsu_id] = {}
        self.client_loss[rsu_id] = loss

    def flush(self):
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
