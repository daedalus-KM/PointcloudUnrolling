import os
import time
import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

def get_fresult(dic_):
    fresult_ = ''
    key_params = dic_.keys()
    for key in sorted(key_params):
        value_str = str(dic_[key])
        if value_str.find('_') >= 0:
            value_str = value_str.replace("_", "-")

        if key == 'resume':
            continue
            
        fresult_ += str(key) + '=' + value_str + '_'
    return fresult_[:-1]

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

@dataclass
class TrainConfig:
    """ training config """

    debug: bool = False  

    att: int = 10
    lr: float = 0.00005 
    b: int = 24
    k: int = 6
    nepoch: int = 100
    nblock: int = 3 
    nconv: int = 3 
    nscale: int = 8
    nheight: int = 128
    feat: int = 0  
    pret: int = 0
    
    run2: int = 0

    def get_ddata(self):
        dataroot = f"{ROOT_DIR}/gen_data/data/"
        return os.path.join(dataroot, "mpi") + '/'
    
    def get_dresult(self):
        fresult = get_fresult(vars(self))
        resroot = f"{ROOT_DIR}/result/"
        dresult = f"{resroot}/ours_{fresult}/"
        os.makedirs(dresult, exist_ok=True)
        return dresult
