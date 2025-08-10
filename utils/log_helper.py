from asyncio.log import logger
import os
import sys

from RSTLNet.utils.excel_helper import write_excel_xlsx

sys.path.append(os.getcwd())
import logging
import numpy as np


def save_epoch(logger, epoch=1, epochs=200, train_mse=0., val_mse=0., val_er=0.):
    train_log_txt = "Epoch:[{}/{}]\t train_loss={:.6}\t val_loss={:.6}\t val_er={:.6}".format(epoch, epochs, train_mse,
                                                                                              val_mse, val_er)
    logger.info(train_log_txt)


def save_epoch_test_result(logger, epoch=1, epochs=200, mse=0., er=0., rmse=0., r2=0.):
    train_log_txt_formatter = "[Test Epoch {epoch}/{epochs}] [MSE] {MSE:.6f} [RMSE] {RMSE:.6f} [R2] {R2:.6f}  [ER] {ER:.6f}\n"
    to_write = train_log_txt_formatter.format(epoch=epoch, epochs=epochs, MSE=mse, RMSE=rmse, R2=r2, ER=er)
    logger.info(to_write)


def save_result(logger, dict_name='', mae=0., mse=0., rmse=0., mape=0., r2=0.):
    train_log_txt_formatter = "[Dict_name] {Dict_name} [MAE] {MAE:.6f} [MSE] {MSE:.6f} [RMSE] {RMSE:.6f} [MAPE] {MAPE:.6f} [R2] {R2:.6f}\n"
    to_write = train_log_txt_formatter.format(Dict_name=dict_name, MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, R2=r2)
    if logger is not None:
        logger.info(to_write)


# def save_to_excel(path,arg,log_file,RMSE,R2,ER):
#     # Model	Seq Len	Pre Len	Missing Ratio	RMSE	R²	Error Ratio	Log	Args	RMSE1	R²1	ER1	RMSE2	R²2	ER2	RMSE3	R²3
#     value = [arg.model,arg.seq_len,arg.pre_len,arg.missing_ratio,np.mean(RMSE),np.mean(R2),np.mean(ER),
#               log_file,str(arg),RMSE[0],R2[0],ER[0],RMSE[1],R2[1],ER[1],RMSE[2],R2[2],ER[2]]
#     write_excel_xlsx(path,[value])
#     logger.info('Test results had save to ' + path)

def save_to_excel(path, arg, log_file, MAE, MSE, RMSE, MAPE, R2, sheet_name_xlsx):
    # Model	Seq Len	Pre Len	Missing Ratio	RMSE	R²	Error Ratio	Log	Args	RMSE1	R²1	ER1	RMSE2	R²2	ER2	RMSE3	R²3
    value = [arg.model, arg.seq_len, arg.missing_ratio, np.mean(MAE), np.mean(MSE), np.mean(RMSE), np.mean(MAPE), np.mean(R2),
             log_file, str(arg), MAE[0], MAE[1], MAE[2], MAE[3], MAE[4], MAE[5], MAE[6], MAE[7], MAE[8], MAE[9],
             MSE[0], MSE[1], MSE[2], MSE[3], MSE[4], MSE[5], MSE[6], MSE[7], MSE[8], MSE[9],
             RMSE[0], RMSE[1], RMSE[2], RMSE[3], RMSE[4], RMSE[5], RMSE[6], RMSE[7], RMSE[8], RMSE[9],
             MAPE[0], MAPE[1], MAPE[2], MAPE[3], MAPE[4], MAPE[5], MAPE[6], MAPE[7], MAPE[8], MAPE[9],
             R2[0], R2[1], R2[2], R2[3], R2[4], R2[5], R2[6], R2[7], R2[8], R2[9],
             arg.loss_func]
    write_excel_xlsx(path, sheet_name_xlsx, [value])
    logger.info('Test results had save to ' + path)


def save_to_excel_detail_rmse(path, arg, detail_rmse, sheet_name_xlsx):
    for d in detail_rmse:
        value1 = [arg.model, d[0][0], d[1][0], d[2][0], d[3][0], d[4][0], d[5][0], d[6][0], d[7][0], d[8][0], d[9][0]]
        write_excel_xlsx(path, sheet_name_xlsx, [value1])
        value2 = [arg.model, d[0][1], d[1][1], d[2][1], d[3][1], d[4][1], d[5][1], d[6][1], d[7][1], d[8][1], d[9][1]]
        write_excel_xlsx(path, sheet_name_xlsx, [value2])
        value3 = [arg.model, d[0][2], d[1][2], d[2][2], d[3][2], d[4][2], d[5][2], d[6][2], d[7][2], d[8][2], d[9][2]]
        write_excel_xlsx(path, sheet_name_xlsx, [value3])
    logger.info('Test results had save to ' + path)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "x")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
