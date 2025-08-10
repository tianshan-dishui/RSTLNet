import argparse
from configs import config

def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=config['model'], help='model name')
    parser.add_argument('--prediction_model', default=config['prediction_model'], help='prediction_model name')
    parser.add_argument('--imputer_name', default=config['imputer_name'], help='imputer_name')
    parser.add_argument('--loss_func', default=config['loss_func'], help='loss_func')
    parser.add_argument('--dataset', default=config['dataset'], help='chose dataset', choices=['geant', 'abilene'])
    parser.add_argument('--gpu', default=config['gpu'],type=int, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
    parser.add_argument('--device', default=config['device'],type=int, help='device')
    parser.add_argument('--cpu', default=config['cpu'],type=int, help='cpu cores')
    parser.add_argument('--epochs', default=config['epochs'],type=int, help='epochs')
    parser.add_argument('--batch_size', '--bs', default=config["batch_size"],type=int, help='batch_size')
    parser.add_argument('--learning_rate', '--lr', default=config["learning_rate"], help='learning_rate')
    parser.add_argument('--seq_len', default=config["seq_len"], type=int, help='input history length')
    parser.add_argument('--pre_len', default=config["pre_len"], type=int, help='prediction length')
    parser.add_argument('--sw_step', default=config["sw_step"], type=int, help='sw_step length')
    parser.add_argument('--train_rate', default=config["train_rate"], help='')
    parser.add_argument('--test_rate', default=config["test_rate"], help='')
    parser.add_argument('--dropout', default=config["dropout"], help='dropout rate')
    parser.add_argument('--lw', default=config["lw"], help='loss1 weight')
    parser.add_argument('--missing_ratio','--ms', default=config["missing_ratio"],type=float, help='missing rate')
    parser.add_argument('--early_stop','--es',type=int, default=config["early_stop"], help='early stop patient epochs')
    parser.add_argument('--rounds', default=config["rounds"], help='rounds')
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to train the model")
    parser.add_argument("--do-eval", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    return parser.parse_args()
