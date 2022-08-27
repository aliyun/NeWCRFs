import random

percentage = 0.02
def separate_train_validate(l,percentage=70):
    sz = len(l)
    cut = int(percentage/100 * sz)  # 80% of the list
    shuffled_l = l.copy()
    random.shuffle(shuffled_l)
    l2 = shuffled_l[:cut]  # first 80% of shuffled list
    l3 = shuffled_l[cut:]  # last 20% of shuffled list
    return l2,l2
import glob
from pathlib import Path
data_path = Path('/home/eposner/Repositories/NeWCRFs/Data')



for folder in data_path.rglob('Train'):
    print(folder)
    rgbs = sorted(list(Path(folder).rglob('*FrameB*.png')))
    depth = sorted(list(Path(folder).rglob('*Depth*.png')))
files_list = []
for i, _ in enumerate(rgbs):
    # remove linebreak from a current name
    # linebreak is the last character of each line
    files_list.append(str(rgbs[i]) + ' ' + str(depth[i]) + '\n')
train, test = separate_train_validate(files_list, percentage=percentage)

with open(r'data_splits/colsim_train_files_with_gt.txt', 'w') as fp:
    fp.writelines(train)
with open(r'data_splits/colsim_test_files_with_gt.txt', 'w') as fp:
    fp.writelines(test)

# with open(r'data_splits/splits_colsim.txt', 'r') as fp:
#         filenames = fp.readlines()
#
# print('read')