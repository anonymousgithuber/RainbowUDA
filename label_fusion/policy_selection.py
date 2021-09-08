import numpy as np
import os
TEACHERS_LIST = ['CBST-GTA5', 'MRKLD-GTA5', 'CAG-GTA5', 'R-MRNet-GTA5', 'SAC-GTA5', 'DACS-GTA5', 'ProDA-GTA5']
NUM_CLASSES = 19
CERTAINTY_DIR = '/home/user/Code/RainbowUDA/train_deeplabv3+/new_certainty_per_class/'
EXTRACTION_SAVE_DIR = '/home/user/Code/RainbowUDA/label_fusion/extraction_list/rand_3.npy'

class_certainty_best = np.zeros(NUM_CLASSES, dtype=np.float32)
class_best = np.zeros(NUM_CLASSES, dtype=np.uint8)
best = 0.
for i in range(NUM_CLASSES):
    for index, item in enumerate(TEACHERS_LIST):
        certainty_arr = np.load(os.path.join(CERTAINTY_DIR, item + '.npy'))
        print("Classs:{:d}:{:.9f}".format(i, certainty_arr[i]))
        if certainty_arr[i] != np.NAN and certainty_arr[i] >= best:
            best = certainty_arr[i]
            class_best[i] = index # teacher model
            class_certainty_best[i] = certainty_arr[i] # certainty value
    best = 0.

extraction_list = [ [idx, value] for idx, value in enumerate(class_best)]
np.save(EXTRACTION_SAVE_DIR, extraction_list)

print(extraction_list)