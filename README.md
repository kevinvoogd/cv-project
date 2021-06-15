# Computer Vision project.
##### Author 1 - Guru Deep Singh (5312558)   
##### Author 2 - Kevin Luis Voogd (4682688)

## Inroduction
In this repository we aim to develop a Computer Vision by Deep Learning project for the course CS4245 of TU Delft. In this repository, we use the findings of the paper *[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)*. In the paper the authors propose a new general-purpose backbone structure for computer vision.

## Adapted Model
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)

## Repository Contents
This repository is self-contained.
##### Python Scripts
- Train a Swin Transformer model with parameters mentioned in another config file. [[Train]](https://github.com/kevinvoogd/cv-project/blob/main/train.py) [[Config]](https://github.com/kevinvoogd/cv-project/blob/main/config.py)
- Testing the trained model. [[Test]](https://github.com/kevinvoogd/cv-project/blob/main/test.py)
- Plot the graph for Taining and Test loss. [[Plot-Graph]](https://github.com/kevinvoogd/cv-project/blob/main/plot_loss.py)

Some other python scripts to make the approach more modular.

##### Datasets
- [Imagenet Subset](https://github.com/kevinvoogd/cv-project/tree/main/datasets)

![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/prediction%201_43_epoch_model.JPG)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/prediction%202_43_epoch_model.JPG)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/prediction%203_43_epoch_model.JPG)

The Repository also provides multiple [pre-trained models](https://github.com/kevinvoogd/cv-project/tree/main/models) which we trained ourselves from scratch. We have also provided all the loss file generated while training and testing.[Pkl](https://github.com/kevinvoogd/cv-project/tree/main/pkl_files)

## How to use this repository
- Clone the Repository
##### Training
- Setup the [[Config]](https://github.com/kevinvoogd/cv-project/blob/main/config.py) (If you want to try something else)
- Execute [train.py](https://github.com/kevinvoogd/cv-project/blob/main/train.py)

##### Testing
- In [test.py](https://github.com/kevinvoogd/cv-project/blob/main/test.py) select the type of model you want to test. This will generate the pkl file for the test which could eventually be plotted. 
- 
Note- If you want to visualize the predicted image against ground truth. Set the variabel "Self.showdata" as True in Config.py file.

##### Plotting the losses
- Select the pkl file in the script [plot_loss](https://github.com/kevinvoogd/cv-project/blob/main/plot_loss.py). This will create the graph for the training and testing which are saved in pkl_files directory.


## Results
##### Training epochs vs Accuracy
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/Epochs%20vs%20Accuracy.JPG)

##### Training loss (10 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/training_10_epoch.JPG)

##### Training loss (30 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/training_30_epoch.JPG)

##### Training loss (100 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/training_100_epoch.JPG)

##### Testing Accuracy (10 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/10_epoch.JPG)

##### Testing Accuracy (30 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/30_epoch.JPG)

##### Testing Accuracy (100 Epochs)
![alt text](https://github.com/kevinvoogd/cv-project/blob/main/pictures/100_epoch.JPG)
