# LocFuse （Pytorch）

## LocFuse overview
The following is our dual-modal descriptor generation network LocFuse, which is used for the place recognition task.

![LocFuse overview](Locfuse.PNG )

## Our Demo

Successful matching candidates are marked in green, and failed ones are marked in red. 

![Our Demo](demo_locfuse.gif)

## Preparation

To begin, download the four sequences from the RobotCar dataset as a demonstration through [this link](https://example.com/path/to/file.zip](https://1drv.ms/u/c/15abfec70c0a221d/EUYaPfXNdjlClfQocYsrEb0BbB59MK_Tgy_YnZuTvRDAYg)). If you require all the sequences, please refer to the benchmark established in the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad?tab=readme-ov-file). 

After downloading the four sequences, place the "RobotCar_samples" folder in the root directory of the project.

Once the dataset files are in place, run the generate_training_tuples.py and generating_test_sets.py scripts in the "generating_queries" folder to obtain the .pickle files required for training and testing.

```
cd generating_queries
python generate_training_tuples.py
python generating_test_sets.py
```
## Train

During the training process, we typically use multi-GPU training by default, employing the nn.DataParallel. However, if you only have a single GPU available, you'll need to modify the corresponding parts of the code accordingly. The trained parameters from our experiments are saved in the file ["weights2/tmp_9_22_best/weight_best.pth"](https://github.com/Bryan-ZhengRui/LocFuse/tree/main/weights2/tmp_9_22_best). The training command is as follows:

```
python train_qua.py
```

## Test

During the testing process, we default to importing the parameters from the file "weights2/tmp_9_22_best/weight_best.pth". You can modify the import path of the parameters as needed. The testing command is as follows:

```
python test.py
```
