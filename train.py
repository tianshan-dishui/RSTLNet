import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append(os.getcwd())
from args import parse_args
from RSTLNet.restruction_train import restruction_train


def main():
    args = parse_args()
    if args.model == 'Ours':
        restruction_train.main()
    else:
        print("No this model")


if __name__ == '__main__':
    main()
    # print("ok!!!")
