
import glob
import os
import shutil


# path = '/home/ysd/cyf20/data/CityscapeSnow/train/input'
# path_new = '/home/ysd/cyf20/data/CityscapeSnow/train/input2'
# all_path = glob.glob(f'{path}/*.png')

# for i in all_path:
    
#     # import pdb;pdb.set_trace()
#     name = os.path.basename(i)
#     print(name)

#     name_id = name.split('-')[1]
#     name_id2 = name_id.split('.')[0]
    
#     if int(name_id2)>2000:
#         print(os.path.join(path_new,name))
#         shutil.move(i,os.path.join(path_new,name))

# path = '/home/ysd/cyf20/data/DDN/test/input'
# path = '/home/ysd/cyf20/data/DDN/train/input'
# path = '/home/ysd/cyf20/data/DDN/test/sci_difficult_illu_correct'
# path = '/home/ysd/cyf20/data/DDN/train/sci_difficult_illu_correct'
# path = '/home/ysd/cyf20/data/Haze4K/train/sci_difficult_illu_correct'
# path = '/home/ysd/cyf20/data/Haze4K/test/input'
path = '/home/ysd/cyf20/data/Haze4K/train/input'
# all_path = glob.glob(f'{path}/*.png')
all_path = glob.glob(f'{path}/*.jpg')




for i in all_path:
    # import pdb;pdb.set_trace()
    name = os.path.basename(i)
    # name_id = name.split('_')[0]
    name_id = name.split('.jpg')[0]
    # name_new = name_id + '.jpg'
    name_new = name_id + '.png'

    os.rename(i,os.path.join(path,name_new))




