# sci_illu_extract

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
