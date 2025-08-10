from RSTLNet.model.DLinear import DLinear
from RSTLNet.model.Ours import Ours
from RSTLNet.model.pre_PatchTST import pre_PatchTST



def get_model(args):
    model_name = args.model
    model = None
    if model_name == 'Ours':
        model = Ours(device=args.device, my_adp=args.support, in_seq=args.seq_len, dropout=args.dropout, flow=args.flow)

    return model.to(args.device)

def get_prediction_model(args):
    model_name = args.prediction_model
    model = None
    if model_name == 'PatchTST':
        model = pre_PatchTST(args)
    if model_name == 'DLinear':
        model = DLinear(args)

    return model.to(args.device)
