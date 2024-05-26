
# This is a code for ddpm with torch 
from functools import partial
from utils.script_utils import get_transform ,cycle
import torch
from argparse import ArgumentParser 
from Model.diffusion import GaussianDiffusion , generate_cosine_schedule ,generate_linear_schedule
from Model.Unet import UNet
from torch.utils.data import DataLoader
from torchvision import datasets
from Model.dataset import DiffusionDataset ,Diffusion_dataset_collate
import torch.optim as optim 
import numpy as np
import gin 
@gin.configurable("options")
def get_options(dataset="default", batch_size=1,
                lr=0.001, betas=(0.9, 0.999),
                network="unet",isload=False,channels=128,
                schedule="linear",num_timesteps=1000,
                schedule_low=1e-4 , schedule_high= 0.02 , 
                img_h = 32 ,img_w = 32 ,optimizer ="adam"
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
        "optimizer" : optimizer
    }

def parse_arguments():
    parser = ArgumentParser(description='Training script:')
    parser.add_argument('--gin_config', default='', type=str, help='Path to the gin configuration file')
    parser.add_argument('--pre_weights' ,default='',type=str ,help='pre-training weight of path ')
    parser.add_argument('--use_labels' ,default=False,type=bool ,help='is use labels  ')
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
    weight_decay = 0 
    channel = options.get("channels")
    num_timesteps = options.get("num_timesteps")
    input_shape = (options.get("img_h") , options.get("img_w"))
    # betas = options.get("betas")
    schedule = options.get("schedule")
    schedule_low = options.get("schedule_low")
    schedule_high = options.get("schedule_high")
    batch_size = options.get("batch_size")
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
    diffusion_model.to(device=device)
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
        raise("isload 错误类型")
    
    lr = options.get("lr")
    # 定义优化器 
    optimizer_dict = {'adam' : optim.Adam(diffusion_model.parameters() ,lr = lr ,betas=(0.5,0.999),weight_decay=weight_decay),
                 'adamw' : optim.AdamW(diffusion_model.parameters() ,lr = lr ,betas=(0.5,0.999),weight_decay=weight_decay),
                 }
    optimizer = optimizer_dict[options.get("optimizer")]
    
    # 数据集path 
    # data_path = ''
    # train_sampler = None 
    # train_dataset = DiffusionDataset(data_path,input_shape=input_shape)
    # # dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,
    # #                         num_workers=4,pin_memory=True,drop_last=True,
    # #                         collate_fn=Diffusion_dataset_collate,sampler=train_sampler)
    
    
    train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=get_transform(),
        )

    test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=get_transform(),
        )

    train_loader = cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        ))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
        

    
    total_loss = 0.0 
    
    for epoch in  range(1,1000+1) : 
        
        # 开启训练     
        diffusion_model.train() 
        
        x , y = next(train_loader) 
        x = x.to(device) 
        y = y.to(device)
        if args.use_labels : 
            
            diffusion_loss = diffusion_model(x,y)
        else : 
            diffusion_loss = diffusion_model(x)
            
            
        total_loss +=diffusion_loss.item()
        optimizer.zero_grad()
        diffusion_loss.backward()
        optimizer.step()
        diffusion_model.update_ema()
        print(f" loss : {diffusion_loss.item()}")
        
        # 测试
        if epoch % 100 == 0 : 
            test_loss = 0 
            with torch.no_grad() : 
                diffusion_model.eval()
                
                for x , y in test_loader : 
                    x = x.to(device) 
                    y = y.to(device) 
                    
                    if args.use_labels : 
                        loss = diffusion_model(x,y) 
                    else :  
                        loss = diffusion_model(x)
                    test_loss +=loss.item()
            
            if args.use_labels:
                samples = diffusion_model.sample(10, device, y=torch.arange(10, device=device))
            else : 
                samples = diffusion_model.sample(10, device)
                
            samples =  ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy() 
            
            test_loss /= len(test_loader) 
            total_loss /= 1000
            

        
        
        
        
        
        
        
    

if __name__ == "__main__" : 
    
    train()