import os
import os.path
import sys
import logging
import copy
import time
import torch
from server.server_base import Server_base

from server.server_lander import Server_LANDER
from server.server_glfc import Server_GLFC
from server.server_lga import Server_LGA
from server.server_target import Server_TARGET
from server.server_fot import Server_FOT
from server.server_fedprotip import Server_FedProTIP


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')
    print(args)
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        
        
        
def _train(args):
    if args["method"] == "baseline":
        server = Server_base(args)
    if args["method"] == "LANDER":
        server = Server_LANDER(args)
    if args["method"] == "FedGPM":
        server = Server_FedProTIP(args)
    if args["method"] == 'GLFC':
        server = Server_GLFC(args)
    if args["method"] == "LGA":
        server = Server_LGA(args)
    if args["method"] == "TARGET":
        server =Server_TARGET(args)
    if args["method"] == "FOT":
        server = Server_FOT(args)

    server.train()
    
