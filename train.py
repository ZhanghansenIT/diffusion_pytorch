
# This is a code for ddpm with torch 
from functools import partial
import torch
from argparse import ArgumentParser 
from Model.diffusion import GaussianDiffusion , generate_cosine_schedule ,generate_linear_schedule
from Model.Unet import UNet


import numpy as np
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
    parser.add_argument('--pre_weights' ,default='',type=str ,help='pre-training weight of path ')
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
    
    diffusion_arg = {}
    
    channel = options.get("channels")
    num_timesteps = options.get("num_timesteps")
    input_shape = tuple(options.get("img_h") , options.get("img_w"))
    betas = options.get("betas")
    schedule = options.get("schedule")
    schedule_low = options.get("schedule_low")
    schedule_high = options.get("schedule_high")
    if schedule == "cosine" : 
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        ) 
    pre_weights_path  = args.pre_weights 
    
    diffusion_model = GaussianDiffusion(UNet(3, channel), input_shape, 3, betas=betas)
    if isinstance(isload,bool) : 
        if isload :
            # 加载预训练权重     
            model_dict = diffusion_model.state_dict( )
            pre_train_dict = torch.load(pre_weights_path,map_location=device)
            pre_train_dict = { k : v for k , v in pre_train_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pre_train_dict)
            diffusion_model.load_state_dict(model_dict)
    else: 
        # 不是 bool 
        raise("不能加载")
    
    

if __name__ == "__main__" : 
    
    train()