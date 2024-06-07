# LocFuse （Pytorch）

## LocFuse overview
The following is our dual-modal descriptor generation network LocFuse, which is used for the place recognition task.

![LocFuse overview](Locfuse.PNG )

## Our Demo

Successful matching candidates are marked in green, and failed ones are marked in red. 

![Our Demo](demo_locfuse.gif)

## Preparation

To begin, download the four sequences from the RobotCar dataset as a demonstration through [this link](https://example.com/path/to/file.zip). If you require all the sequences, please refer to the benchmark established in the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad?tab=readme-ov-file). 

After downloading the four sequences, place the "RobotCar_samples" folder in the root directory of the project.

Once the dataset files are in place, run the generate_training_tuples.py and generating_test_sets.py scripts in the "generating_queries" folder to obtain the .pickle files required for training and testing.

`cd generating_queries` 
`python generate_training_tuples.py`
