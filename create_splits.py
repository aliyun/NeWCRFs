import random
overfit = False

percentage = 0.02 if overfit else 70
def separate_train_validate(l,percentage=70):
    sz = len(l)
    cut = int(percentage/100 * sz)  # 80% of the list
    shuffled_l = l.copy()
    random.shuffle(shuffled_l)
    l2 = shuffled_l[:cut]  # first 80% of shuffled list
    l3 = shuffled_l[cut:]  # last 20% of shuffled list
    if overfit:
        return l2,l2
    return l2,l3
import glob
from pathlib import Path
data_path = Path('/home/eposner/Repositories/NeWCRFs_erez/NeWCRFs/Data')


train_val_files_list = []
test_files_list = []

for folder in data_path.rglob('Train'):
    print(folder)
    rgbs = sorted(list(Path(folder).rglob('FrameB*.png')))
    depth = sorted(list(Path(folder).rglob('Depth*.png')))
    for i, _ in enumerate(rgbs):
        # remove linebreak from a current name
        # linebreak is the last character of each line
        train_val_files_list.append(str(rgbs[i]) + ' ' + str(depth[i]) + '\n')
for folder in data_path.rglob('Test'):
    print(folder)
    rgbs = sorted(list(Path(folder).rglob('FrameB*.png')))
    # depth = sorted(list(Path(folder).rglob('Depth*.png')))
    for i, _ in enumerate(rgbs):
        # remove linebreak from a current name
        # linebreak is the last character of each line
        test_files_list.append(str(rgbs[i]) + '\n')
train, val = separate_train_validate(train_val_files_list, percentage=percentage)

with open(r'data_splits/colsim_train_files_with_gt.txt', 'w') as fp:
    fp.writelines(train)
with open(r'data_splits/colsim_valid_files_with_gt.txt', 'w') as fp:
    fp.writelines(val)
with open(r'data_splits/colsim_test_files_with_gt.txt', 'w') as fp:
    fp.writelines(test_files_list)

# with open(r'data_splits/splits_colsim.txt', 'r') as fp:
#         filenames = fp.readlines()
#
# print('read')