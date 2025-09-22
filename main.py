import json
import argparse
from trainer import train
import os
import wandb 

def main():
    dir = "./run"
    args = setup_parser().parse_args()
    param = load_json(args.config)  
    args = vars(args)   
    args.update(param)  
    args["save_dir"] = os.path.join(dir, args["exp_name"])

    if args["wandb"]:
        os.environ["WANDB_INIT_TIMEOUT"] = "300"

        run = wandb.init(
            dir=dir,
            project=args["wandb_proj"],
            name=f'{args["wandb_name"]}-{args["seed"]}',
            reinit=True
        )
        wandb.config.update(args) 

    train(args)

    if args["wandb"]:
        run.finish()


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of FedMPQ algorthm.')
    parser.add_argument('--config', type=str, default='configs/cifar100/fedprotip.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
