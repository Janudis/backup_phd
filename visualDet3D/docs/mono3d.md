# Mono3D

![mono3d_image](mono3d.png)

## Training Schedule

```bash
# copy mono 3D example config
cd config
cp Yolo3D_example $CONFIG_FILE.py

## Modify config path
nano $CONFIG_FILE.py
cd ..

## Compute image database and anchors mean/std
# You can run ./launcher/det_precompute.sh without arguments to see helper documents
./launcher/det_precompute.sh config/$CONFIG_FILE.py train
./launcher/det_precompute.sh config/$CONFIG_FILE.py test # only for upload testing
./launchers/det_precompute.sh config/config.py train

## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
./launcher/train.sh  --config/$CONFIG_FILE.py 0 $experiment_name # validation goes along
./launchers/train.sh config/config.py 0 fine_tune_nuscenes_full

## produce validation/test result # we only support single GPU testing
# You can run ./launcher/eval.sh without arguments to see helper documents
./launcher/eval.sh --config/$CONFIG_FILE.py 0 $CHECKPOINT_PATH validation # test on validation split
./launcher/eval.sh --config/$CONFIG_FILE.py 0 $CHECKPOINT_PATH test # test on test split
./launchers/eval.sh --config/config.py 0 worksdirs/Mono3D/checkpoint/ test
./launchers/eval.sh config/config.py 0 /home/dimitris/PhD/PhD/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth validation

```

## Testing on Lyft dataset without retraining
![mono3dgif](mono3d_lyft.gif)
