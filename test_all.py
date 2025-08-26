import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel

from multi_read_data import MemoryFriendlyLoader
from datasets.all_light import AllLight
import yaml


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/medium',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/medium', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument("--test_input_path", type=str, default='raindrop')
parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

with open(os.path.join("./datasets/all_light.yml"), "r") as f:
    config = yaml.safe_load(f)

new_config = dict2namespace(config)

new_config.data.data_name = args.test_set
new_config.path.real_weather = args.test_input_path


alllight_func = AllLight(new_config)
train_loader, val_loader = alllight_func.get_loaders(parse_patches=False)



def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        # for loader in [train_loader, val_loader]:
        # for _, (input, image_name) in enumerate(test_queue):
        for _, (x, y) in enumerate(val_loader): # train_loader
            x = Variable(x[:, :3], volatile=True).cuda()
            image_name = y[0].split('\\')[-1]  #.split('.')[0]
            print(image_name)
            i, r = model(x)
            # print(image_name)
            u_name = '%s.png' % (image_name)
            # print(u_name)
            print('processing {}'.format(u_name))
            u_name = os.path.basename(u_name)
            u_path = save_path + '/' + u_name
            # save_images(r, u_path)
            # exit()
            save_images(i, u_path)
            # exit()


if __name__ == '__main__':
    main()

'''

########## LOL-v1 ##########

CUDA_VISIBLE_DEVICES=6 python test_all.py --save_path=../../data/LOL-v1/our485/sci_difficult_illu/ --model=./weights/difficult.pt
OK

CUDA_VISIBLE_DEVICES=6 python test_all.py  --save_path=../../data/LOL-v1/eval15/sci_difficult_illu/ --model=./weights/difficult.pt

########## lol_v2_real ##########

CUDA_VISIBLE_DEVICES=6 python test_all.py --save_path=../../data/LOL-v2/Real_captured/Train/sci_difficult_illu/  --model=./weights/difficult.pt

OK

CUDA_VISIBLE_DEVICES=6 python test_all.py  --save_path=../../data/LOL-v2/Real_captured/Test/sci_difficult_illu/ --model=./weights/difficult.pt

########## lol_v2_syn ##########


CUDA_VISIBLE_DEVICES=6 python test_all.py --save_path=../../data/LOL-v2/Synthetic/Train/sci_difficult_illu/ --model=./weights/difficult.pt

OK

CUDA_VISIBLE_DEVICES=6 python test_all.py  --save_path=../../data/LOL-v2/Synthetic/Test/sci_difficult_illu/ --model=./weights/difficult.pt

########## SDSD-Indoor ##########

CUDA_VISIBLE_DEVICES=7 python test_all.py --save_path=../../data/SDSD/indoor_static_np/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## SDSD-Outdoor ##########

CUDA_VISIBLE_DEVICES=6 python test_all.py --save_path=../../data/SDSD/outdoor_static_np/sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## SID ##########

CUDA_VISIBLE_DEVICES=7 python test_all.py --save_path=../../data/sid_processed/sci_difficult_illu_correct --model=./weights/difficult.pt


########## SMID ##########

CUDA_VISIBLE_DEVICES=7 python test_all.py  --save_path=../../data/smid/sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## LOL-Blur ##########

CUDA_VISIBLE_DEVICES=7 python test_all.py  --save_path=../../data/LOL-Blur/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt

CUDA_VISIBLE_DEVICES=7 python test_all.py  --save_path=../../data/LOL-Blur/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## lol_deblur_in_LDR ##########

CUDA_VISIBLE_DEVICES=5 python test_all.py  --save_path=../../data/lol_deblur_in_LDR/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## lol_deblur_in_MIMO ##########

CUDA_VISIBLE_DEVICES=5 python test_all.py  --save_path=../../data/lol_deblur_in_MIMO/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## fog ##########
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/CityscapeFog/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/CityscapeFog/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## raindrop ##########
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/Raindrop/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/Raindrop/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## rain ##########
CUDA_VISIBLE_DEVICES=3 python test_all.py  --save_path=../../data/DDN/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=3 python test_all.py  --save_path=../../data/DDN/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt
########## snow  ##########
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/CityscapeSnow/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=3 python test_all.py  --save_path=../../data/CityscapeSnow/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## haze ##########
CUDA_VISIBLE_DEVICES=3 python test_all.py  --save_path=../../data/Haze4K/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=3 python test_all.py  --save_path=../../data/Haze4K/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## snow11k  ##########
CUDA_VISIBLE_DEVICES=5 python test_all.py  --save_path=../../data/snow11k/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=5 python test_all.py  --save_path=../../data/snow11k/train/sci_difficult_illu_correct/ --model=./weights/difficult.pt

########## LOL-Blur-noise ##########
CUDA_VISIBLE_DEVICES=6 python test_all.py  --save_path=../../data/LOL-Blur/test/lol_blur_noise_sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=5 python test_all.py  --save_path=../../data/LOL-Blur/train/lol_blur_noise_sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## raindrop_snow ##########
CUDA_VISIBLE_DEVICES=4 python test_all.py  --save_path=../../data/Raindrop_snow/test/snowS_sci_difficult_illu_correct/ --model=./weights/difficult.pt
CUDA_VISIBLE_DEVICES=0 python test_all.py  --save_path=../../data/Raindrop_snow/train/snowS_sci_difficult_illu_correct/ --model=./weights/difficult.pt


########## real_snow ##########
CUDA_VISIBLE_DEVICES=5 python test_all.py --test_input_path ../../data/real_weather/real_snow/test/input --test_set real_weather --save_path=../../data/real_weather/real_snow/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt 
########## real_raindrop ##########
CUDA_VISIBLE_DEVICES=5 python test_all.py --test_input_path ../../data/realblur-iphone14pro  --test_set real_weather   --save_path=../../data/real_weather/realblur/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
########## real_fog ##########
CUDA_VISIBLE_DEVICES=4 python test_all.py  --test_input_path ../../data/realhaze-RESIDE/Unannotated\ Real-world\ Hazy\ Images --test_set real_weather  --save_path=../../data/real_weather/realhaze/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
########## real_blur ##########
CUDA_VISIBLE_DEVICES=5 python test_all.py  --test_input_path ../../data/realfog-iphone14pro --test_set real_weather  --save_path=../../data/real_weather/realfog/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt
########## real_haze ##########
CUDA_VISIBLE_DEVICES=4 python test_all.py  --test_input_path ../../data/realraindrop-iphone14pro --test_set real_weather  --save_path=../../data/real_weather/realraindrop/test/sci_difficult_illu_correct/ --model=./weights/difficult.pt

'''