# Rainbow UDA: Combining Domain Adaptive Models for Semantic Segmentation Tasks

## Training Step and Script
### File Structure
```
Projec_Root_DIR/
├── weights/
│   ├── weights/
│   │   ├── synthia/
│   │   └── gta5/
│   └── weights.zip
├── RainbowUDA/
└── Dataset/
    ├── GTA5
    │   ├── images
    │   ├── labels
    ├── SYNTHIA
    │   └── RAND_CITYSCAPES
    └── cityscapes
        ├── gtFine
        ├── leftImg8bit
        └── pseudo_label
            ├── GTA5
            │   ├── CAG-GTA5
            │   │   ├── aachen_000000_000019_leftImg8bit.npz
            │   │   ├── aachen_000000_000019_leftImg8bit.png
            │   │   ├── aachen_000001_000019_leftImg8bit.npz
            │   │   ├── aachen_000001_000019_leftImg8bit.png
            │   │   ├──...
            │   ├── CBST-GTA5
            │   ├── DACS-GTA5
            │   ├── MRKLD-GTA5
            │   ├── ProDA-GTA5
            │   ├── R-MRNet-GTA5
            │   └── SAC-GTA5
            └── SYNTHIA
                ├── CAG-SYNTHIA
                ├── CBST-SYNTHIA
                ├── DACS-SYNTHIA
                ├── MRKLD-SYNTHIA
                ├── ProDA-SYNTHIA
                ├── R-MRNet-SYNTHIA
                └── SAC-SYNTHIA
```
### Preparation 
* Generate 7 teachers PL and certainty on `train` (ex. ProDA-GTA5)
* Generate 7 teachers distilled PL and certainty on `train` (ex. ProDA-GTA5-DT) 
* Generate 7 teachers class certainty on `val` (ex. certainty_per_class/ProDA-GTA5/class_certainty.npy)
```
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 test.py --utils class_certainty --restore-from ./snapshot_distilled_student/snapshot_ProDA-GTA5/proda_gta5_best_model.pth --result-dir ./certainty_per_class/ProDA-GTA5
```
* Generate policy based on class certainty (SETTING in FILES) (ex. extraction_list_stage1.npy)
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 policy_selection.py
```

### Training Baseline
* EnD-H-PL
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 label_fusion.py --fusion-mode majority --result-dir /home/user/Dataset/cityscapes/pseudo_label/GTA5/majority --teachers-list CBST-GTA5 MRKLD-GTA5 R-MRNet-GTA5  CAG-GTA5 SAC-GTA5 DACS-GTA5 ProDA-GTA5 --num-teachers 7
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 train.py --set majority --snapshot-dir ./snapshot_majority_gta5 --result-dir ./result/majority_gta5 
```
* EnD
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 label_fusion.py --fusion-mode average --result-dir /home/user/Dataset/cityscapes/pseudo_label/GTA5/average --teachers-list CBST-GTA5 MRKLD-GTA5 R-MRNet-GTA5  CAG-GTA5 SAC-GTA5 DACS-GTA5 ProDA-GTA5 --num-teachers 7
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 train_EnD_baseline.py --set average --snapshot-dir ./snapshot_EnD_gta5 --result-dir ./result/EnD_gta5
```
* EnD-S
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 label_fusion.py --fusion-mode average --result-dir /home/user/Dataset/cityscapes/pseudo_label/GTA5/average_calibrated --teachers-list CBST-GTA5-DT MRKLD-GTA5-DT R-MRNet-GTA5-DT CAG-GTA5-DT SAC-GTA5-DT DACS-GTA5-DT ProDA-GTA5-DT --num-teachers 7
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 train_EnD_baseline.py --set average_calibrated --snapshot-dir ./snapshot_EnD_calibrated_gta5 --result-dir ./result/EnD_calibrated_gta5
```
* EnD-PL
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 label_fusion.py --fusion-mode certainty --result-dir /home/user/Dataset/cityscapes/pseudo_label/GTA5/certainty_calibrated --teachers-list CBST-GTA5-DT MRKLD-GTA5-DT R-MRNet-GTA5-DT DACS-GTA5-DT CAG-GTA5-DT SAC-GTA5-DT ProDA-GTA5-DT --num-teachers 7
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 train.py --set certainty_calibrated --snapshot-dir ./snapshot_certainty_calibrated_gta5 --result-dir ./result/certainty_calibrated_gta5 --batch-size 8 
```

### Training Ours
```
cd /home/user/Code/RainbowUDA/label_fusion
python3 label_fusion.py --fusion-mode channel --result-dir /home/user/Dataset/cityscapes/pseudo_label/GTA5/channel_stage_1 --extraction-list-dir ./extraction_list_stage1.npy --teachers-list CBST-GTA5 MRKLD-GTA5 R-MRNet-GTA5 CAG-GTA5 SAC-GTA5 DACS-GTA5 ProDA-GTA5 --num-teachers 7
cd /home/user/Code/RainbowUDA/train_deeplabv3+
python3 train.py --set channel_stage_1 --snapshot-dir ./snapshot_ours_stage1 --result-dir ./result/DUMMY --class-balance --often-balance --batch-size 8
```