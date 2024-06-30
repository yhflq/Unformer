import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import argparse
import time

from model import underFormer
parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/LSUI/test/image/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/LSUI/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./LSUI/checkpoints1/un_Former/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')
args = parser.parse_args()



def load_checkpoint(model, weights):
    checkpoint = torch.load(weights) 
    try:
        model.load_state_dict(checkpoint["state_dict"]) 
    except:
        state_dict = checkpoint["state_dict"]  
        new_state_dict = OrderedDict() 
        for k, v in state_dict.items(): 
            name = k[7:]  
            new_state_dict[name] = v  
        model.load_state_dict(new_state_dict)  

inp_dir = args.input_dir  
out_dir = args.result_dir


os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))



if len(files) == 0:  
    raise Exception(f"No files found at {inp_dir}")  

model = underFormer(dim=48)
model.cuda()  

load_checkpoint(model, args.weights) 
model.eval() 


mul = 16  
index = 0  
total_time = 0  # Initialize total time
for file_ in files: 
    img = Image.open(file_).convert('RGB') 
    input_ = TF.to_tensor(img).unsqueeze(0).cuda() 
    h, w = input_.shape[2], input_.shape[3]  
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul 
    padh = H - h if h % mul != 0 else 0  
    padw = W - w if w % mul != 0 else 0  
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect') 

    

    start_time = time.time()  
    with torch.no_grad():  
        restored = model(input_) 

    end_time = time.time()  
    inference_time = end_time - start_time  
    total_time += inference_time 


    index += 1 
    print(f'{index}/{len(files)} - Inference Time: {inference_time:.4f} seconds')  


average_time = total_time / len(files)  
print(f"Average Inference Time: {average_time:.4f} seconds")  
print(f"Files processed from {inp_dir}")
