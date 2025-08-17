import csv
train_txt_file = '/home/ubuntu/data1/kinetics400/k400_train.txt'
val_txt_file = '/home/ubuntu/data1/kinetics400/k400_val.txt'
with open(train_txt_file) as f:
    lines = f.readlines()

# train_csv_file = 'train.csv'
# with open(train_csv_file, 'w') as f:
#     csvwriter = csv.writer(f)
#     for line in lines:
#         path, _, label = line.split(' ')
#         path = 'train_256/' + path
#         line = path + ' ' + label[:-1]
#         csvwriter.writerow([line])

# with open(val_txt_file) as f:
#     lines = f.readlines()

# val_csv_file = 'val.csv'
# with open(val_csv_file, 'w') as f:
#     csvwriter = csv.writer(f)
#     for line in lines:
#         path, _, label = line.split(' ')
#         path = 'val_256/' + path
#         line = path + ' ' + label[:-1]
#         csvwriter.writerow([line])

train_csv_file = 'train_video_list.txt'
with open(train_csv_file, 'w') as f:
    for line in lines:
        path, _, label = line.split(' ')
        line = path + ' ' + label[:-1]
        f.write(line)
        f.write('\n')

with open(val_txt_file) as f:
    lines = f.readlines()

val_csv_file = 'val_video_list.txt'
with open(val_csv_file, 'w') as f:
    for line in lines:
        path, _, label = line.split(' ')
        line = path + ' ' + label[:-1]
        f.write(line)
        f.write('\n')