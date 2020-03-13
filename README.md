# Hyperplane Arrangements of Trained ConvNets Are Biased

This repository contains the source code to reproduce the experiments of the paper "Hyperplane Arrangements of Trained ConvNets Are Biased". The code is based on Pytorch.

## Requirements

To install the dependencies for the project, run

  ```bash
    pip install -r requirements.txt
  ```

## Introduction

In this work, we take a geometrical perspective and look for statistical bias in the weights of trained convolutional networks, in terms of hyperplane arrangements induced by convolutional layers with ReLU activations. Notably, for networks combining linear (affine) layers with piece-wise linear activations, hyperplane arrangements define the function computed by the network and characterize how data is transformed non-linearly by the model.

Our main message is summarized as follows.

*Hyperplane arrangements of many layers of trained convolutional networks exhibit strong regularities, that emerge from training and correlate with learning. Furthermore, for low-complexity datasets, layers presenting biased hyperplane arrangements are critical to the performance of the network -- our measure correlates with the notion of critical layers introduced in the recent intriguing work of Zhang et al.```[```[1](https://arxiv.org/abs/1902.01996)```]```. That means, when the bias is not observed, the corresponding layers' weights can be reset to their value at initialization without considerable loss in performance.*

We refer the reader to the paper for detailed explanation and motivation for our methodology.

## Training models

To train a model, run

  ```bash
  train.py [-h] [--arch ARCH] [--dataset DATASET]
                [--subsample-classes SUBSAMPLE_CLASSES]
                [--class-sample-seed CLASS_SAMPLE_SEED] [--noise NOISE]
                [--noise-seed NOISE_SEED] [--augmentation] [--pretrained]
                [--split SPLIT] [--stratified] [--split-seed SPLIT_SEED]
                [--shuffle] [--evaluate] [--evaluate-train] [--cuda]
                [--workers WORKERS] [--data-path DATA_PATH] [--epochs EPOCHS]
                [--start-epoch START_EPOCH] [--batch-size BATCH_SIZE]
                [--optimizer OPTIMIZER] [--lr LR] [--lr-step LR_STEP]
                [--lr-decay LR_DECAY] [--weight-decay WEIGHT_DECAY]
                [--momentum MOMENTUM] [--models-path MODELS_PATH]
                [--seed SEED] [--tb-logdir TB_LOGDIR]
                [--snapshot-every SNAPSHOT_EVERY]
                [--snapshot-all-until SNAPSHOT_ALL_UNTIL]
                [--resume-from RESUME_FROM] [--kill-plateaus] [--train-acc]
                [--unnormalize] [--log LOG]

    Model training/finetuning.

    optional arguments:
    -h, --help            show this help message and exit
    --arch ARCH           Network architecture to be trained. Run without this
                          option to see a list of all supported archs.
    --dataset DATASET     Dataset to train the network on. Run without this
                          option to see a list of supported datasets.
    --subsample-classes SUBSAMPLE_CLASSES
                          Subsample only SUBSAMPLE_CLASSES classes from DATASET.
                          If set to 0 (default) all classes of DATASET are used.
    --class-sample-seed CLASS_SAMPLE_SEED
                          Numpy random seed used for sampling classes.
    --noise NOISE         Ratio of corrupted labels, in [0., 1.]. Set to -1 to
                          enable pixel shuffle in place of label noise.
    --noise-seed NOISE_SEED
                          Numpy seed for corrupting labels.
    --augmentation        Enable data augmentation.
    --pretrained          Load pretrained architecture on ImageNet from model
                          zoo.
    --split SPLIT         Validation split size. The resutling split is class-
                          unbalanced [default=0].
    --stratified          Validation split with equal balance of samples per
                          class.
    --split-seed SPLIT_SEED
                          Seed used for shuffling the data before making the
                          validation split.
    --shuffle             Shuffle the training data before making the validation
                          split [default=False].
    --evaluate            Evaluate model on the validation set.
    --evaluate-train      Evaluate model on the training set.
    --cuda                Enable GPU support.
    --workers WORKERS, -j WORKERS
                          Number of parallel data processing jobs.
    --data-path DATA_PATH
                          Path to local ImageNet folder.
    --epochs EPOCHS       The number of epochs used for training [default = 20].
    --start-epoch START_EPOCH
                          Starting epoch for training.
    --batch-size BATCH_SIZE
                          The minibatch size for training [default = 128].
    --optimizer OPTIMIZER
                          Supported optimizers: sgd, adam [default = sgd].
    --lr LR               The base learning rate for SGD optimization [default =
                          0.001].
    --lr-step LR_STEP     The step size (# iterations) of the learning rate
                          decay [default = off].
    --lr-decay LR_DECAY   The decay factor of the learning rate decay [default =
                          off].
    --weight-decay WEIGHT_DECAY
                          The weight decay coefficient [default = 1e-4].
    --momentum MOMENTUM   The momentum coefficient for SGD [default = 0.9].
    --models-path MODELS_PATH
                          The dirname where to store/load models [default =
                          './models'].
    --seed SEED           Pytorch PRNG seed.
    --tb-logdir TB_LOGDIR
                          The tensorboard log folder [default =
                          './tensorboard'].
    --snapshot-every SNAPSHOT_EVERY
                          Snapshot the model state every E epochs [default = 0].
    --snapshot-all-until SNAPSHOT_ALL_UNTIL
                          Optional. Snapshot every epoch until the specified
                          one, then snapshot according to the --snapshot-every
                          argument.
    --resume-from RESUME_FROM
                          Path to a model snapshot [default = None].
    --kill-plateaus       Quit training if the model validation accuracy
                          plateaus in the first 10 epochs.
    --train-acc           (Optional) compute train accuracy and report it during
                          training.
    --unnormalize         Disable data normalization and represent instead pixel
                          values in [0,1]. [Default = False].
    --log LOG             Logfile name [default = 'train.log'].
  ```
  
To see a list of supported architectures run ```train.py``` with no ```--arch``` argument, e.g.

  ```bash
    python train.py --dataset imagenet
  ```
  
Similarly, to see a list of supported datasets, run ```train.py``` with no ```--dataset``` argument, e.g.

  ```bash
    python train.py --arch vgg19
  ```

Training supports logging to ```tensorboard```.
  
### Example -- Train VGG19 on ImageNet

  To train VGG19 on ImageNet on GPU(s) for 40 epochs, with base learning rate 0.1 and batch size 128, run
  ```bash
    python train.py --arch vgg19 --dataset imagenet --cuda --data-path /path/to/imagenet --epochs 40 --lr 0.01 --batch-size 128
  ```


## Computing Projection Statistics

When training a model, several snapshots of the model at different epochs can be saved. For each snapshot, our projection statistics can be computed by running

  ```bash
    compute_projections.py [-h] [--arch ARCH] [--dataset DATASET]
                              [--load-from LOAD_FROM] [--init-from INIT_FROM]
                              [--pretrained] [--cuda] [--normalize]
                              [--results RESULTS] [--log LOG] [--seed SEED]

  Compute projection statistics.

  optional arguments:
    -h, --help            show this help message and exit
    --arch ARCH           Network architecture to be trained. Run without this
                          option to see a list of all available pretrained
                          archs.
    --dataset DATASET     Dataset used to train the model. Used to specify the
                          number of classes of the prediction layer of the
                          model.
    --load-from LOAD_FROM
                          Load trained network from file.
    --init-from INIT_FROM
                          (Optional) network initialization. Specify a model snapshot to
                          measure distance from initialization for each layer.
    --pretrained          Load pretrained architecture on ImageNet
                          [default=False].
    --cuda                Enable GPU support.
    --normalize           Normalize projections.
    --results RESULTS     Path to store results [default = './results'].
    --log LOG             Logfile name [default = 'positive_orthant.log'].
    --seed SEED           Pytorch seed. (Optional)If using --load-from, specify
                          the seed the model was trained with.
  ```
If no model snapshots are available, it is possible to compute the statistics for off-the-shelf pretrained Pytorch models.

  ```bash
    python compute_projections.py --arch alexnet --dataset imagenet --cuda --normalize --pretrained
  ```
  
Finally, when no model snapshots or ```--pretrained``` are specified, the statistics for one random initialization of the specified architecture can be computed:

  ```bash
    python compute_projections.py --arch alexnet --dataset imagenet --cuda --normalize
  ```
  
For each snapshot or pretrained model, the compute statistics will be stored as a JSON file, which can later be loaded for plotting.

### Example
Compute projection statistics from a model snasphot of AlexNet trained on Imagenet:
  ```bash
    python compute_projections.py --arch alexnet --dataset imagenet --cuda --normalize --load-from /path/to/model/snapshot
  ```

## Weight Reinitialization

To reproduce the experiments of Zhang et al., we reimplemented the methodology described in section 2 of ```[```[1](https://arxiv.org/abs/1902.01996)```]```.

Given a snapshot of a trained network and a list of snapshots of initial weights to use for weight reinitialization, the drop in validation accuracy can be computed by running

  ```bash
    reinit_layers.py [-h] [--arch ARCH] [--dataset DATASET]
                        [--load-from LOAD_FROM] [--inits-from INITS_FROM]
                        [--cuda] [--results RESULTS] [--log LOG] [--seed SEED]
                        [--workers WORKERS] [--data-path DATA_PATH]
                        [--batch-size BATCH_SIZE] [--rand]

    Reinit network weights and compute drop in performance.

    optional arguments:
    -h, --help            show this help message and exit
    --arch ARCH           Network architecture. Run without this option to see a
                          list of all available pretrained archs.
    --dataset DATASET     Dataset used to train the model. Used to specify the
                          number of classes of the prediction layer of the
                          model.
    --load-from LOAD_FROM
                          Load trained network from file.
    --inits-from INITS_FROM
                          Network initialization. Specify a model snapshot to
                          measure distance from initialization for each layer.
    --cuda                Enable GPU support.
    --results RESULTS     Path to store results [default = './results'].
    --log LOG             Logfile name [default = 'reinit_weights.log'].
    --seed SEED           Pytorch seed. (Optional)If using --load-from, specify
                          the seed the model was trained with.
    --workers WORKERS, -j WORKERS
                          Number of parallel data processing jobs.
    --data-path DATA_PATH
                          Path to local dataset folder.
    --batch-size BATCH_SIZE
                          The minibatch size for training [default = 128].
    --rand                Perform weight randomization test.
  ```
  
The list of weights ```INITS_FROM``` to use for reinitialization must be specified as a plaintext list of paths to model snapshots -- one per line -- with the file ending with an empty line.
  
If the flag ```--rand``` is specified, the drop in accuracy is computed also for weights sampled from a (independent) random initialization.

All results are stored as JSON for easy post-processing (e.g. plotting heatmaps).

### Example
Weight Reinitialization for VGG19 trained on ImageNet:
   ```bash
     python reinit_weights.py --arch vgg19 --dataset imagenet --data-path /path/to/imagenet --load-from /path/to/trained/model/snapshot --inits-from /path/to/file/list/of/initializations.txt --rand --batch-size 128 --cuda
   ```
   
## References

 1. "[Are All Layers Created Equal?](https://arxiv.org/abs/1902.01996)." Chiyuan Zhang and Samy Bengio and Yoram Singer. 2019. *arXiv preprint 1902.01996*.
