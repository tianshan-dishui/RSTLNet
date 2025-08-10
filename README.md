# RSTLNet: A Robust Spatio-Temporal Learning Framework for Network Traffic Matrix Imputation
## Introduction
This repo is the implementation of "RSTLNet: A Robust Spatio-Temporal Learning Framework for Network Traffic Matrix Imputation" (IEEE Transactions on Neural Networks and Learning Systems, Under review).
## Dataset
Two publicly available datasets are utilized to validate the proposed imputation method, namely the Abilene and GÉANT datasets. They provide the statistical traffic volume data of the real network traffic trace from the American Research and Education Network (Abilene) and the Europe Research and Education Network (GÉANT) .
Topology  | Nodes  | Flows | Links | Interval | Horizon | Records
 ---- | ----- | ------  | ------| ------| ------| ------
 Abilene  | 12 | 144  | 15 | 5 min | 6 months | 48096 
 GÉANT   | 23 | 529  | 38 | 15 min | 4 months | 10772 
 ## Framework
 ![image](https://github.com/tianshan-dishui/RSTLNet/blob/main/picture/framework.png)
 ## Abstract
RSTLNet integrates network-specific prior knowledge, such as network topology and routing configurations, into a weighted adjacency matrix, enhancing spatio-temporal correlation modeling. By segmenting TM sequences and applying patch-wise spatial learning, RSTLNet captures dynamic spatial dependencies and adapts to network changes. Additionally, a data contribution index is introduced to adjust the influence of neighboring nodes based on data completeness, reducing the impact of unreliable information in aggregation. Experiments on the Abilene and Geant datasets demonstrate that RSTLNet achieves superior imputation accuracy, training efficiency, and parameter efficiency compared to state-of-the-art models. 
## Getting Started
### Mask Generation
utils/data_help.py
``` python
mask_generation(dataset='abilene', ratio=0.35, counts=3)
```
### Config
configs.py
``` python
config = Config(
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    gpu=0,
    cpu=os.cpu_count(),
    model='Ours',
    prediction_model='DLinear',
    imputer_name='mean',
    loss_func='MSE',
    dataset='abilene',  # abilene or geant
    epochs=200,
    batch_size=32,
    learning_rate=0.0001,
    seq_len=48,  # complete length
    pre_len=48,  # number of timestamps to predict (If prediction is needed)
    sw_step=1,
    rounds=3,
    train_rate=0.6,
    test_rate=0.2,
    dropout=0.1,
    missing_ratio=0.35,
    early_stop=15,
    lw=0.5,
)
```
### Training
utils/data_help.py
``` Bash
python train.py
```
