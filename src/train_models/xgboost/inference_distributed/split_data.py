import yaml
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

data_path = config['files']['data_path'] 

distributed_nodes = 4

if __name__ == "__main__":
    ######## Load data
    t1 = time.time()
    test = pd.read_parquet(f'{data_path}/stage12_test')  
    print(test.shape)
    print(f"load data took {time.time() - t1} s")

    ######## split data
    t1 = time.time()
    indexs = [i for i in range(distributed_nodes)]
    step = int(len(test)/distributed_nodes)
    tests = []
    for i in range(distributed_nodes):
        if i<distributed_nodes-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])
        
    for i in range(len(tests)):
        tests[i].to_parquet(f"{data_path}/stage12_test_{i}.parquet")

    print(f"totally took {time.time() -very_start} s")
