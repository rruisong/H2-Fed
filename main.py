#!/usr/bin/env python
import argparse
import os
import random
import copy
import torch
import numpy as np
import json
from json import JSONEncoder
import pickle
from tqdm import trange
from colorama import Fore

from clients.agent import FedClient
from aggregation_layers.rsu import FedServer
from aggregation_layers.oem_cloud import OemCloud
from aggregation_layers.traffic_cloud import TrafficCloud
from post_processing.recorder import Recorder
from config.batch_sim_config import sim_config
from config.double_agg_layer import *
from config.single_agg_layer import *

json_types = (list, dict, str, int, float, bool, type(None))
DATA_SET = "mnist"

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def load_data(data_set=DATA_SET):
    train_data_dir = os.path.join('data', data_set, 'train')
    test_data_dir = os.path.join('data', data_set, 'test')

    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        train_data.update(cdata['user_data'])
    client_id_list = cdata['users']
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    test_iid_data = {'x': [], 'y': []}
    for user in test_data.keys():
        test_iid_data['x'].extend(test_data[user]['x'])
        test_iid_data['y'].extend(test_data[user]['y'])

    return test_iid_data, test_data, train_data, client_id_list


def pre_train(dir_model, name_model):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    test_iid_data, test_data, train_data, client_id_list = load_data(data_set=DATA_SET)
    test_vehicle_list = client_id_list[:10]
    pre_train_data = {'x': [], 'y': []}
    for client_id in test_vehicle_list:
        pre_train_data['x'].extend(train_data[client_id]['x'])
        pre_train_data['y'].extend(train_data[client_id]['y'])
    oem_cloud = OemCloud('OEM', 50, 0)
    oem_cloud.load_train_data(pre_train_data)
    oem_cloud.load_test_data(test_iid_data)
    print("OEM starts pre-training")
    state_dict, loss, n_data, recorder = oem_cloud.train()

    torch.save(state_dict, os.path.join(dir_model, name_model))
    return state_dict


def base_train(dir_model, name_model):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    test_iid_data, test_data, train_data, client_id_list = load_data(data_set=DATA_SET)
    test_vehicle_list = client_id_list[10:]
    base_train_data = {'x': [], 'y': []}
    for client_id in test_vehicle_list:
        base_train_data['x'].extend(train_data[client_id]['x'])
        base_train_data['y'].extend(train_data[client_id]['y'])
    oem_cloud = OemCloud('OEM', 60, 0)
    oem_cloud.load_train_data(base_train_data)
    oem_cloud.load_test_data(test_iid_data)
    print("OEM starts centralized training")
    state_dict, loss, n_data, recorder = oem_cloud.train()
    with open(os.path.join(dir_model, name_model), "w") as jsfile:
        json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


def fed_run(GAR,  # global aggregation rsu_round
            LAR,  # local aggregation rsu_round
            CSR,  # connection success ratio
            SCD,  # stable connection duration
            FSR,  # full-task success ratio
            mu,  # core parameters M in framework
            epoch_init,  # init epoch wrt. FSR
            res_dir,  # directory of results
            rsu_config,  # rsu configuration data
            label,  # label for result data
            save_model,
            i_seed=0 # seed
            ):
    np.random.seed(i_seed)
    torch.manual_seed(i_seed)
    random.seed(i_seed)

    # data preprocessing and postprocessing
    test_iid_data, test_data, train_data, client_id_list = load_data(data_set=DATA_SET)
    recorder = Recorder(rsu_config)

    # Initialize rsu and clients for testing
    rsu_dict = {}
    global_rsu_list = []
    global_client_dict = {}
    for rsu_id in rsu_config.keys():
        client_dict = {}
        client_list = []
        for client_id in rsu_config[rsu_id]:
            b = np.random.binomial(np.ones(1).astype(int), FSR)
            if b:
                epoch = epoch_init[1]
            else:
                epoch = epoch_init[0]
            client_dict[client_id] = FedClient(client_id, epoch, copy.deepcopy(mu))
            client_dict[client_id].rsu_id = rsu_id
            global_client_dict[client_id] = {}
            global_client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_train_data(train_data[client_id])
            client_dict[client_id].load_test_data(test_iid_data)
            client_list.append(client_id)

        rsu = FedServer(client_list, rsu_id)
        rsu_dict[rsu_id] = rsu
        rsu.load_test_data(test_iid_data)
        global_rsu_list.append(rsu_id)

    # Initializing the temporary variables
    rsu_model_state, n_data, avg_loss = {}, 0, 0

    # Initialize cloud
    cloud = TrafficCloud(global_rsu_list)
    cloud.load_test_data(test_iid_data)
    cloud.model.load_state_dict(torch.load('pre_train/model'))

    accuracy = cloud.test()
    recorder.res['server']['iid_accuracy'].append(accuracy)
    recorder.res['server']['train_loss'].append(2)
    cloud.flush()

    # Initialize all RSUs
    for rsu_id in rsu_dict.keys():
        rsu_dict[rsu_id].receive_global_weight(cloud.model.state_dict())

    # Create dir for model saving
    model_dir = ('models_' + label)
    model_save_dir = os.path.join('models_' + res_dir, model_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Start training
    bar_desc = ('Accuracy of pre-trained model is: %.4f' % accuracy)
    t = trange(GAR, desc=bar_desc, leave=True)
    for global_round in t:

        # 1: all rsu get the model from global server
        cloud.select_clients(1)
        for rsu_id in rsu_dict.keys():
            rsu = rsu_dict[rsu_id]
            if rsu_id in cloud.selected_rsu_ids:
                rsu.receive_global_weight(cloud.model.state_dict())
            else:
                print(rsu_id)
            for rsu_round in range(LAR):

                # All clients keep updating their models
                # Only the selected clients will be aggregated, i.e. self.selected_client_ids
                if (global_round + 1) * (rsu_round + 1) % SCD == 0:
                    rsu.select_clients(CSR)

                # Start training in each client
                for client_id in rsu.selected_client_ids:
                    # Download models before training
                    global_client_dict[client_id].local_update(rsu.model.state_dict(), rsu.global_model.state_dict())
                    current_client = global_client_dict[client_id]
                    state_dict, loss, n_data = current_client.train()
                    rsu.receive(client_id, state_dict, n_data, loss)
                    # Record the training results in each Client
                    recorder.res['clients'][client_id]['rsu_id'] = rsu_id
                    recorder.res['clients'][client_id]['iid_accuracy'] = None
                    recorder.res['clients'][client_id]['train_loss'].append(loss)

                rsu_model_state, avg_loss, n_data = rsu.aggregate()
                rsu.flush()
                accuracy = rsu.test()

                # Record the training results in each RSU
                recorder.res['rsu'][rsu_id]['iid_accuracy'].append(accuracy)
                recorder.res['rsu'][rsu_id]['train_loss'].append(avg_loss)

            if rsu_id in cloud.selected_rsu_ids:
                cloud.receive(rsu_id, rsu_model_state, n_data, avg_loss)
            else:
                print(rsu_id)
        cloud_model_state, avg_loss, n_data = cloud.aggregate()

        # Save the model
        if save_model:
            torch.save(cloud_model_state, os.path.join(model_save_dir, 'global_round_' + str(global_round)))

        accuracy = cloud.test()
        cloud.flush()
        # Record the training results in Cloud
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        t.set_description(label +
                          '| Cloud accuracy: %.4f' % accuracy +
                          '| Cloud Avg loss: %4f' % avg_loss)

        if global_round % 5 == 0:
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            with open(os.path.join(res_dir, label), "w") as jsfile:
                json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)
    print(Fore.GREEN, 'End training ' + label + ': results in ' + res_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--GAR', type=int, default=60,help='Global Aggregation Round')
    parser.add_argument('-l', '--LAR', type=int, default=10, help='Local Aggregation Round')
    parser.add_argument('-c', '--CSR', type=float, default=0.1, help='Connection Success Ratio')
    parser.add_argument('-s', '--SCD', type=int, default=1, help='Stable Connection Duration')
    parser.add_argument('-f', '--FSR', type=int, default=1, help='Full-task Success Ratio')
    parser.add_argument('-m', '--mu', nargs='+', help='Core parameters M in framework')
    parser.add_argument('-ei', '--epoch_init', nargs='+', help='Init epoch wrt. FSR')
    parser.add_argument('-rd', '--res_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('-rc', '--rsu_config', type=str, default='rsu_config_NonIID_RSU', help='RSU configuration dict data in ./config/*')
    parser.add_argument('-la', '--label', type=str, default='H2Fed', help='Label for result data')
    parser.add_argument('-b', '--batch', type=bool, default=False, help='If the existing scenarios are batched run')
    parser.add_argument('-sm', '--save_model', type=bool, default=False, help='If the model should be saved')
    parser.add_argument('-bt', '--base_train', type=bool, default=False, help='If the centralized training should be implemented first')
    parser.add_argument('-pt', '--pre_train', type=bool, default=False, help='If the pre-training should be implemented first')
    parser.add_argument('-rr', '--res_root', type=str, default='results', help='Root directory of the results')

    opt = parser.parse_args()

    if opt.mu is None:
        opt.mu = [0.001, 0.001]
    if opt.epoch_init is None:
        opt.epoch_init = [10, 10]

    BASE_TRAIN = opt.base_train
    PRE_TRAIN = opt.pre_train
    res_root = opt.res_root
    save_model = opt.save_model

    # Pre-training
    if PRE_TRAIN:
        dir_model = 'pre_train'
        name_model = 'model'
        pre_train(dir_model, name_model)

    # Centralized training
    if BASE_TRAIN:
        dir_model = 'base_train'
        name_model = 'model'
        base_train(dir_model, name_model)

    if opt.batch:
        for scenario in sim_config:
            for param in scenario["param_list"]:
                # Federated training
                fed_run(GAR=param['GAR'],
                        LAR=param['LAR'],
                        CSR=param['CSR'],
                        SCD=param['SCD'],
                        FSR=param['FSR'],
                        mu=param['mu'],
                        epoch_init=param['epoch_init'],
                        res_dir=os.path.join(res_root, scenario["res_dir"]),
                        rsu_config=param["config"],
                        label=param["label"],
                        save_model=param['save_model'])
    else:
        fed_run(GAR=opt.GAR,
                LAR=opt.LAR,
                CSR=opt.CSR,
                SCD=opt.SCD,
                FSR=opt.FSR,
                mu={'mu1': float(opt.mu[0]), 'mu2': float(opt.mu[1])},
                epoch_init=[int(x) for x in opt.epoch_init],
                res_dir=os.path.join(res_root, opt.res_dir),
                rsu_config=eval(opt.rsu_config),
                label=opt.label,
                save_model=opt.save_model)


if __name__ == "__main__":
    main()