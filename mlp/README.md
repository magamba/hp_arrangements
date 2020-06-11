# Hyperplane Arrangements of Trained ConvNets Are Biased

This repository contains the source code to reproduce the experiments on fully connected networks, reported in the supplemental material to the paper. The code is based on Pytorch 1.5.

## Requirements

To install the dependencies for the project, run

  ```bash
    pip install -r requirements.txt
  ```

Please note that the requirements are slightly different from those reported for the experiments on ConvNets. Particularly, pytorch with version at least 1.5 is required.

## Training models

To train a MLP to reproduce the results reported in the supplemental material, run:

  ```bash
    python train.py --arch mlp_bn \
                    --dataset cifar10 \
                    --augmentation \
                    --shuffle \
                    --cuda \
                    --workers 4 \
                    --epochs 300 \
                    --batch-size 128 \
                    --lr 0.01 \
                    --lr-step 100 \
                    --lr-decay 0.1 \
                    --weight-decay 5e-4 \
                    --snapshot-every 10 \
                    --snapshot-all-until 10 \
                    --train-acc
  ```

Training supports logging to ```tensorboard```.
  
## Computing Projection Statistics

When training a model, several snapshots of the model at different epochs can be saved. For each snapshot, the projection statistics can be computed by running

  ```bash
  python compute_projections.py --arch 'mlp_bn' \
                                --dataset 'cifar10' \
                                --cuda \
                                --load-from "/path/to/model/checkpoint.pt" \
                                --normalize \
                                --with-linear \
                                --init-from "/path/to/init/checkpoint.pt" 
  ```
  
For each checkpoint, the computed statistics will be stored as a JSON file, which can later be loaded for plotting.
