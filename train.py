
# This is a code for ddpm with torch 
from functools import partial
import torch
from argparse import ArgumentParser 

import gin 
@gin.configurable("options")
def get_options(dataset="default", batch_size=1,
                lr=0.001, betas=(0.9, 0.999),
                network="unet",isload=False,channels=128,
                schedule="linear",num_timesteps=1000,
                schedule_low=1e-4 , schedule_high= 0.02 , 
                img_h = 32 ,img_w = 32 
                ):
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "lr": lr,
        "betas": betas,
        "network" : network,
        "isload" : isload , 
        "channels" : channels , 
        "schedule" : schedule , 
        "num_timesteps" : num_timesteps , 
        "schedule_low" : schedule_low , 
        "schedule_high" : schedule_high , 
        "img_h" : img_h , 
        "img_w" : img_w,
    }

def parse_arguments():
    parser = ArgumentParser(description='Training script:')
    parser.add_argument('--gin_config', default='', type=str, help='Path to the gin configuration file')
    return parser.parse_args()






def train() : 
    args = parse_arguments()
    gin.parse_config_file(args.gin_config)  # 加载配置文件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = get_options()
    
    
    get_noise_predict_network = options.get("network")
    print(f"{get_noise_predict_network}")
    
    if get_noise_predict_network == "U-net" : 
        # 创建 Unet结构 
        pass 
    isload = options.get("isload")
  
    if isinstance(isload,bool) : 
        if isload :
            # 加载  
            pass 
        else : 
            pass 
    else: 
        # 不是 bool 
        pass 

if __name__ == "__main__" : 
    
    train()