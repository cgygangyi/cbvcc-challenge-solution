import os

train_path = '/cc_data/AIM/codes/image4video/data/kinetics700/train_256'
val_path = '/cc_data/AIM/codes/image4video/data/kinetics700/val_256'

label_map = {}
label = 0
with open('/cc_data/AIM/codes/image4video/data/kinetics700/train_video_list.txt', 'w') as f:
    dirs = os.listdir(train_path)
    dirs = sorted(dirs)
    for dir in dirs:
        label_map[dir] = label
        slabel = str(label)
        vdir = os.path.join(train_path, dir)
        vnames = os.listdir(vdir)
        for name in vnames:
            video_relative_path = os.path.join(dir, name)
            output = '{} {}\n'.format(video_relative_path, slabel)
            f.write(output)
        label += 1

label_map = {}
label = 0
with open('/cc_data/AIM/codes/image4video/data/kinetics700/val_video_list.txt', 'w') as f:
    dirs = os.listdir(val_path)
    dirs = sorted(dirs)
    for dir in dirs:
        label_map[dir] = label
        slabel = str(label)
        vdir = os.path.join(val_path, dir)
        vnames = os.listdir(vdir)
        for name in vnames:
            video_relative_path = os.path.join(dir, name)
            output = '{} {}\n'.format(video_relative_path, slabel)
            f.write(output)
        label += 1

