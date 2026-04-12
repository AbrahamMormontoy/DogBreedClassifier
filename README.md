# Image Classification Projects

This repository contains two image-classification workflows:

- `CNN_Implementation`: a ResNet-18 dog-breed classifier
- `Logistic Regression`: a baseline with logistic regression

## Requirements

- torch 
- torchvision 
- pandas 
- matplotlib 
- pillow 
- numpy 
- scikit-learn 
- opencv-python

To install of all of requirement you can use the following command:

```bash
pip install torch torchvision pandas matplotlib pillow numpy scikit-learn opencv-python
```

## Main Structure for the CNN

- `CNN_Implementation/dataset/`: dataset with all the breeds
- `CNN_Implementation/best_model.pth`: saved CNN after training
- `CNN_Implementation/logs/`: CSV metrics files
- `CNN_Implementation/plots/`: generated plots
- `CNN_Implementation/images/`: images to predict dog breed
- `CNN_Implementation/utility/`: clean the dataset in selected_breeds
- `CNN_Implementation/predictions/`: predictions png about the animal images in `predictionimages`
- `CNN_Implementation/predictionimages/`: png to be predicted by `predict_single_image.py`

## How To Use The CNN Project

### 1. Dataset
Put the dataset under `CNN_Implementation/dataset/<breed_name>/`. Make sure to follow this structure as the model takes the breed name from the name of the folder that carry the images. 

### 2. Utility scripts

Trim each breed folder to a fixed number of images:

```bash
cd CNN_Implementation
python utility/trim_breeds.py
```

Reduces every breed folder to exactly 150 images by randomly removing extras

Rename breed folders by removing the `n1234-` prefix:

```bash
cd CNN_Implementation
python utility/rename_breeds.py
```

Before: selected_breeds/n02085620-Chihuahua/ 
After:  selected_breeds/Chihuahua/

### 3. Train the main CNN model

```bash
cd CNN_Implementation
python -m src.train
```

This are the defaults inputs `--epochs 15 --lr 0.001`
```bash
cd CNN_Implementation
python -m src.train
```

What it does:

- loads images from `dataset`
- builds a pretrained ResNet-18 backbone
- saves the best checkpoint in `best_model.pth`
- gives the result of the train and the precision of the model

### 4. Predict on one image or a folder

```bash
cd CNN_Implementation
python -m src.predict_single_image
```

This are the defaults inputs `--input path/to/image_or_folder --checkpoint best_model.pth --breeds-dir dataset --topk 3`

What it does:

- loads the saved checkpoint in `best_model.pth`
- predicts the top-k breeds for a single image or every image in the folder `images`
- prints the probability for each predicted breed
- saves a visualization plot to `predictions/predictions.png`

### 4. Run the training-size comparison experiment

```bash
cd CNN_Implementation
python -m src.predict_and_plot
```

This are the defaults inputs `--epochs 5 --small 15 --large 100`

What it does:

- trains two models using different numbers of training images per class
- evaluates per-class accuracy and mean accuracy
- saves CSV summaries in `logs/`
- saves comparison plots in `plots/`


### Not neccesary but good practice 

```bash
cd CNN_Implementation
python model.py
```

This runs a model sanity check using forward pass to verify that the ResNet-18 head is wired correctly.

## Outputs for the CNN

- `CNN_Implementation/best_model.pth`
- `CNN_Implementation/logs/training_log.csv`
- `CNN_Implementation/plots/training_curves.png`
- `CNN_Implementation/predictions/predictions.png`
- `CNN_Implementation/logs/per_class_accuracy.csv`
- `CNN_Implementation/logs/mean_accuracy.csv`
- `CNN_Implementation/plots/per_class_compare.png`
- `CNN_Implementation/plots/mean_accuracy_curve.png`

Top-1 vs Top-3 Accuracy 
Top-1 accuracy is the percentage of test images where the model's single best guess was the 
correct breed. This is the standard accuracy metric. 
Top-3 accuracy is the percentage of test images where the correct breed appeared somewhere in 
the model's top 3 predictions. This is the headline metric for our project.  
Top-3 will always be higher than Top-1. 



## Main structure for the Logistic Regression

- `Logistic Regression/main.py`: get the classification report
- `Logistic Regression/predict_single.py`: predict a single or varios image in the folder predictions
- `Logistic Regression/resize_dataset.py`: fix the 
- `Milestone1 Logistic Regression/image_loader.py`:
- `Milestone1 Logistic Regression/logisticRegression/logistic_regression.py`:

## How To Use the Logistic Regression

### 1. Dataset

Put the dataset under `Logistic Regression/dataset/<breed_name>/`. Make sure to follow this structure as the model takes the breed name from the name of the folder that carry the images. 

### 2. Train the main Logistic Regression model

```bash
cd "Logistic Regression"
python src/main.py
```

What it does:

- loads images from `dataset`
- gives the result of the train and the accuracy of the model
- gives a classification report for the three breeds


### 3. Predict on one image or a folder

```bash
cd "Logistic Regression"
python src/predict_single_image.py 
```

What it does:

- predicts the top-k breeds for a single image or every image in the folder `predictions`
- prints the probability for each predicted breed

## Outputs for the Logistic regression
All the outputs are in the terminal using report with SKLEARN