from config.double_agg_layer import *
from config.single_agg_layer import *

GLOBAL_COMMUNICATION_ROUND = 60

sim_config = [
    {
        'scenario_name': 'Non-IID_RSU_Layer',
        'res_dir': 'Non-IID_RSU_Layer',
        'param_list': [
            {  # FedAvg
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 1,
                'mu': {'mu1': 0, 'mu2': 0},
                'epoch_init': [2, 2],
                'label': "FedAvg",
                "res_dir": "Non-IID_RSU_Layer",
                "config": single_rsu_config,
                "save_model": True
            },
            {  # FedProx
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 1,
                'mu': {'mu1': 0, 'mu2': 0.005},
                'epoch_init': [2, 2],
                'label': "FedProx",
                "res_dir": "Non-IID_RSU_Layer",
                "config": single_rsu_config,
                "save_model": True
            },
            {  # HierFAVG
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 3,
                'mu': {'mu1': 0, 'mu2': 0},
                'epoch_init': [2, 2],
                'label': "HierFAVG",
                "res_dir": "Non-IID_RSU_Layer",
                "config": rsu_config_NonIID_RSU,
                "save_model": True
            },
            {  # H2-Fed
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 3,
                'mu': {'mu1': 0.001, 'mu2': 0.005},
                'epoch_init': [2, 2],
                'label': "H2-Fed",
                "res_dir": "Non-IID_RSU_Layer",
                "config": rsu_config_NonIID_RSU,
                "save_model": True
            },
        ]
    },
    {
        'scenario_name': 'Non-IID_Agent_Layer',
        'res_dir': 'Non-IID_Agent_Layer',
        'param_list': [
            {  # FedAvg
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 1,
                'mu': {'mu1': 0, 'mu2': 0},
                'epoch_init': [2, 2],
                'label': "FedAvg",
                "res_dir": "Non-IID_Agent_Layer",
                "config": single_rsu_config,
                "save_model": True
            },
            {  # FedProx
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 1,
                'mu': {'mu1': 0, 'mu2': 0.001},
                'epoch_init': [2, 2],
                'label': "FedProx",
                "res_dir": "Non-IID_Agent_Layer",
                "config": single_rsu_config,
                "save_model": True
            },
            {  # HierFAVG
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 10,
                'mu': {'mu1': 0, 'mu2': 0},
                'epoch_init': [2, 2],
                'label': "HierFAVG",
                "res_dir": "Non-IID_Agent_Layer",
                "config": rsu_config_NonIID_client,
                "save_model": True
            },
            {  # H2-Fed
                "GAR": GLOBAL_COMMUNICATION_ROUND,
                "CSR": 0.1,
                "SCD": 1,
                "FSR": 1,
                'LAR': 10,
                'mu': {'mu1': 0.001, 'mu2': 0.001},
                'epoch_init': [2, 2],
                'label': "H2-Fed",
                "res_dir": "Non-IID_Agent_Layer",
                "config": rsu_config_NonIID_client,
                "save_model": True
            },
        ]
    }
]