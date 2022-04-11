import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self, rsu_config=None):
        self.res_list = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': []},
                    'rsu': {'iid_accuracy': [], 'train_loss': []},
                    'clients': {'iid_accuracy': [], 'train_loss': []}}
        if rsu_config is not None:
            for rsu_id in list(rsu_config.keys()):
                self.res['rsu'][rsu_id] = {'iid_accuracy': [], 'train_loss': []}
                for client_id in rsu_config[rsu_id]:
                    self.res['clients'][client_id] = {'rsu_id': rsu_id, 'iid_accuracy': [], 'train_loss': []}

    def load(self, filename, label):
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        plt.figure(figsize=(10, 8))
        for i, (res, label) in enumerate(self.res_list):
            if label == 'base_':
                pass
            else:
                p = plt.plot(np.array(res['server']['iid_accuracy']), label=label, alpha=1, linewidth=2)

            rsu_acc_list = np.array([])
            for key in res['rsu'].keys():
                if res['rsu'][key]:
                    rsu_acc_list = np.append(rsu_acc_list, res['rsu'][key]['iid_accuracy'])

        plt.xlabel('Time (s)', size=20)
        plt.ylabel('Accuracy', size=20)
        plt.legend(prop={'size': 20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.grid()


