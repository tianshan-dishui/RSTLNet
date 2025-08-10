import os
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from RSTLNet.utils.early_stop import EarlyStopping

sys.path.append(os.getcwd())
import copy
import sys
import time
from RSTLNet.args import parse_args
from RSTLNet.utils.time_helper import format_sec_tohms, get_datetime_str
import numpy as np
import torch.nn
from tqdm import tqdm
from RSTLNet.utils.metrics import MAE, MSE, R2, RMSE, MAPE
from RSTLNet.utils.log_helper import get_logger, save_epoch, save_result, save_to_excel
from RSTLNet.utils.data_helper import *
from RSTLNet.utils.model_helper import *

np.set_printoptions(threshold=sys.maxsize)
criterion = torch.nn.MSELoss()


def train(args, model, optimizer, criterion, dataloader, device, e):
    train_loss = 0.0
    model.train()
    for x, y in tqdm(dataloader, ncols=80, position=0):
        x = x.to(device)

        if (x.size()[-1] < 3):
            x, mask = x[:, :, :, 0], x[:, :, :, 1]
            x_imputed = x * (1 - mask)
        else:
            x, mask, x_imputed = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]

        x_res = model(x_imputed, mask).squeeze(1)
        x_hat = x_res

        loss = criterion(x_hat * mask, x * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    return train_loss


def validate(args, model, criterion, dataloader, device, e):
    val_loss = 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            if (x.size()[-1] < 3):
                x, mask = x[:, :, :, 0], x[:, :, :, 1]
                x_imputed = x * (1 - mask)
            else:
                x, mask, x_imputed = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]

            x_res = model(x_imputed, mask).squeeze(1)
            x_hat = x_res
            loss = criterion(x_hat * mask, x * mask)
            val_loss += loss.item()
    val_loss = val_loss / len(dataloader)
    return val_loss


def test(args, model, dataloader, num_flows, device, seq_len, logger):
    x_true, x_pred = torch.empty([0, seq_len, num_flows]), torch.empty([0, seq_len, num_flows])
    MASK = torch.empty([0, seq_len, num_flows])
    x_true, x_pred, MASK = x_true.to(device), x_pred.to(device), MASK.to(device)
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            if (x.size()[-1] < 3):
                x, mask = x[:, :, :, 0], x[:, :, :, 1]
                x_imputed = x * (1 - mask)
            else:
                x, mask, x_imputed = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]

            x_res = model(x_imputed, mask).squeeze(1)
            x_hat = x_res
            x_true = torch.cat((x_true, x), 0)
            x_pred = torch.cat((x_pred, x_hat), 0)
            MASK = torch.cat((MASK, mask), 0)
    mae = MAE(x_pred, x_true)
    mse = MSE(x_pred, x_true)
    rmse = RMSE(x_pred, x_true)
    mape = MAPE(x_pred, x_true)
    r2 = R2(x_pred, x_true)
    logger.info("mae  is {}".format(mae))
    logger.info("mse  is {}".format(mse))
    logger.info("rmse  is {}".format(rmse))
    logger.info("mape  is {}".format(mape))
    logger.info("r2  is {}".format(r2))
    return mae, mse, rmse, mape, r2


def main():
    result_xlsx = '/home/zhengkaiyuan/pytorch_test/RSTLNet/TM1.xlsx'
    sheet_name_xlsx = 'Sheet1'
    args = parse_args()
    ts = get_datetime_str()
    log_file = 'logs/' + __file__.split('/')[-1] + "_log_{}_{}.txt".format(args.model, str(ts))
    logger = get_logger(log_file)

    if args.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    device = args.device

    data_path = get_data_path(args.dataset)
    num_nodes, num_flows = get_dataset_nodes(args.dataset)
    args.flow = num_flows
    m_adj = np.load(get_adj_matrix(args.dataset))
    args.m_adj = m_adj
    imputer_name = args.imputer_name

    ALL_MAE, ALL_MSE, ALL_RMSE, ALL_MAPE, ALL_R2 = [], [], [], [], []
    dict_names = []

    dataset_path = os.path.dirname(__file__)
    topo_data_path = args.dataset + '_weight_flow.npy'
    topo_data_path = os.path.join(dataset_path, '../topo', topo_data_path)
    print(topo_data_path)
    topo_data = np.load(topo_data_path)
    print(topo_data.shape)
    topo_data = change_topo_matrix(topo_data)
    topo_data = torch.from_numpy(topo_data).to(torch.float32)
    topo_data = topo_data.to(device)
    print(topo_data.shape)
    print(topo_data)
    args.support = []
    args.support = args.support + [topo_data]

    if args.loss_func == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.loss_func == 'MAE':
        criterion = nn.L1Loss()

    for i in range(1, args.rounds + 1):
        early_stop = EarlyStopping(patience=args.early_stop, logger=logger)
        # dataloader
        dataloader = get_dataloaders(
            data_path=data_path,
            train_rate=args.train_rate,
            test_rate=args.test_rate,
            seq_len=args.seq_len,
            pre_len=args.pre_len,
            sw_step=args.sw_step,
            missing_ratio=args.missing_ratio,
            missing_index=i,
            batch_size=args.batch_size,
            num_workers=args.cpu,
            imputer=imputer_name,
            random=[False, False, False], )

        # model
        model = get_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print(args)

        time_start = time.time()
        train_losses = []
        val_losses = []
        logger.info('Training on ' + args.model)
        ###### train ####
        for e in range(1, args.epochs + 1):
            epoch_train_start_time = time.time()
            train_loss = train(args, model, optimizer, criterion, dataloader['train'], device, e)
            epoch_train_end_time = time.time()
            logger.info(
                'epoch train cost time {}'.format(format_sec_tohms(epoch_train_end_time - epoch_train_start_time)))
            val_loss = validate(args, model, criterion, dataloader['val'], device, e)
            train_losses.append(train_loss)
            save_epoch(logger, e, args.epochs, train_loss, val_loss)
            es, new_high = early_stop(val_loss)
            if new_high:
                early_stop.save_model_dict(copy.deepcopy(model.state_dict()))
                logger.info("*NEW MIN VAL LOSS*")
                tmae, tmse, trmse, tmape, tr2 = test(args, model, dataloader['test'], num_flows, device, args.seq_len,
                                                     logger)
                # save_epoch_test_result(logger,e,args.epochs,tmse,er2,trmse,tr2)
            if es:
                break
        time_end = time.time()
        cost_time = format_sec_tohms(time_end - time_start)
        ts = get_datetime_str()
        dict_name = 'dict/' + model.__class__.__name__ + "_" + args.dataset + "_" + str(
            args.seq_len) + "_" + ts + '_dict.pkl'

        num_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of model parameters: {num_parameters}")

        torch.save(early_stop.get_best_model_dict(), dict_name)
        dict_names.append(dict_name)
        logger.info(ts)
        logger.info(dict_name)
        logger.info(cost_time)
        logger.info(str(args))
        ########## test ###########
        model.load_state_dict(early_stop.get_best_model_dict())
        target_ratio = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        result_matrix = []
        DICT_MAE = []
        DICT_MSE = []
        DICT_RMSE = []
        DICT_MAPE = []
        DICT_R2 = []

        for tr in target_ratio:
            temp_mae = []
            temp_mse = []
            temp_rmse = []
            temp_mape = []
            temp_r2 = []
            for j in range(1, 4):
                test_dataloader = get_dataloaders(
                    data_path=data_path,
                    train_rate=args.train_rate,
                    test_rate=args.test_rate,
                    seq_len=args.seq_len,
                    pre_len=args.pre_len,
                    sw_step=args.sw_step,
                    missing_ratio=tr,
                    missing_index=j,
                    batch_size=args.batch_size,
                    num_workers=args.cpu,
                    imputer=imputer_name,
                    test=True,
                    random=[False, False, False], )
                test_start_time = time.time()
                mae, mse, rmse, mape, r2 = test(args, model, test_dataloader['test'], num_flows, device, args.seq_len,
                                                logger)
                test_end_time = time.time()
                logger.info(
                    'test cost time {}'.format(format_sec_tohms(test_end_time - test_start_time)))
                temp_mae.append(mae.item())
                temp_mse.append(mse.item())
                temp_rmse.append(rmse.item())
                temp_mape.append(mape.item())
                temp_r2.append(r2.item())
            DICT_MAE.append(np.mean(temp_mae))
            DICT_MSE.append(np.mean(temp_mse))
            DICT_RMSE.append(np.mean(temp_rmse))
            DICT_MAPE.append(np.mean(temp_mape))
            DICT_R2.append(np.mean(temp_r2))
            save_result(logger, dict_name, np.mean(temp_mae), np.mean(temp_mse), np.mean(temp_rmse), np.mean(temp_mape),
                        np.mean(temp_r2))
        ALL_MAE.append(DICT_MAE)
        ALL_MSE.append(DICT_MSE)
        ALL_RMSE.append(DICT_RMSE)
        ALL_MAPE.append(DICT_MAPE)
        ALL_R2.append(DICT_R2)

    save_to_excel(result_xlsx, args, log_file, np.mean(ALL_MAE, axis=0), np.mean(ALL_MSE, axis=0),
                  np.mean(ALL_RMSE, axis=0), np.mean(ALL_MAPE, axis=0),
                  np.mean(ALL_R2, axis=0), sheet_name_xlsx)
    logger.info("MEAN_MAE:")
    logger.info(np.mean(ALL_MAE, axis=0))
    logger.info("MEAN_MSE:")
    logger.info(np.mean(ALL_MSE, axis=0))
    logger.info("MEAN_RMSE:")
    logger.info(np.mean(ALL_RMSE, axis=0))
    logger.info("MEAN_MAPE:")
    logger.info(np.mean(ALL_MAPE, axis=0))
    logger.info("MEAN_R2:")
    logger.info(np.mean(ALL_R2, axis=0))
    logger.info(log_file)
    print(dict_names)


if __name__ == '__main__':
    main()
