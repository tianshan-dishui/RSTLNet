import os
import torch


class Config(dict):
    def __init__(self, **kwargs):
        """
        Initialize an instance of this class.

        Args:

        """
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        """
        Sets the value to the value.

        Args:
            key: (str):
            value:
        """
        self[key] = value
        setattr(self, key, value)

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
