**Author:**

Haolong Li (haolong.li@epfl.ch) (352680)

Zhibo Zhao (zhibo.zhao@epfl.ch) (350593)


# Expected File Structure
```python
.
│  .gitattributes
│  .gitignore
│  classify.ipynb # The main pipeline
│  README.md
│  requirements.txt
│  result.csv # The Kaggle submission
│  train.ipynb # The classifier training pipeline, NOT expected to rerun due to lack of training data in the repo.
│
├─data # Placeholder data folder. To generate test results, populate the test folder.
│  ├─ref
│  │      .gitkeep
│  │
│  ├─test
│  │      .gitkeep
│  │
│  └─train
│          .gitkeep
│
└─model
        best_model.pth
        label_to_index.json
```
For reproducing the Kaggle result, run the `classify.ipynb` file.

**Notice that** we were not able to upload the training data & the test data due to file size limits. To generate the Kaggle submission (or any other test predictions), you should populate the `data/test` folder with `.JPG` files.


# Project Outline

In this project we created a coin detection model. For an image containing mutiple coins of different types and values, the model first segments the image to obtain subfigures of separate coins, then each coin figure is fed to a single coin classification model for classification. We finally integrate the results from the subfigures from one image to get the detection result of the image.

## Segmentation
In general this step contains:
- Thresholding;
- Applying Mathematical Morphology;
- Detecting contours from the previous result;
- Identifying circualr contour shapes and append to the result.
  
There are 3 kinds of images: neutral, noisy, hand. For each kind of image, we try 2 different threholding + morphology strategies:
- **neutral bg + rgb thresholding**: 
  - Thresholding: lower bound (rgb) = 0, 0, 0; upper bound (rgb) = 150, 175, 175; 
  - Apply 2 closings (kernel size = 25\*25, 30\*30)
- **noisy bg + rgb thresholding**:
  - Thresholding: lower bound (rgb) = 0, 0, 0; upper bound (rgb) = 185, 180, 160; 
  - Apply 1 closing (kernel size = 11\*11)
- **hand bg + rgb thresholding**:
  - Thresholding: lower bound (rgb) = 0, 0, 0; upper bound (rgb) = 175, 175, 150; 
  - Apply 2 closings (kernel size = 30\*30, 35\*35)
- **adaptive thresholding**
  - Convert image to gray scale;
  - Apply Gaussian blurring (kernel size = 11*11)
  - Apply adaptive thresholding (see params in code)
  - Apply closing:
    - neutral bg: kernel size = 11;
    - noisy bg: kernel size = 20;
    - hand bg: kernel size = 9.

After applying thresholding + morphology, we proceed to the contour stage.
For the contours identfied, we filter out those who size deviates far away from its min enclosing circle's size to ensure we get circular shapes.

In implementation, since we don't know which kind of background the image is, we apply all the above 6 methods, and take the one with the most coins detected.

## Feature Extraction
As described above, for each image containing multiple images, we end up with many subfigures of separate coins, each subfigure containing only one coin. They are supposed to be [x, y, 3] arrays. Each array is then fed to our classifier to identify the coin.

## Classification
We train a model based the resnet pretrained model (on the image 1k dataset) in order to classify **single** coins.

**NOTE:** The training pipeline is not contained in this notebook because it's not possible to upload the whole training data to Moodle. We will however provide the `training.ipynb` in a separate file for the sake of integrity. For the model in this notebook, we will load the model weights from our training pipeline directly.

### Training Data Labeling
We used the labelImg package to label each coin and its position in the image. For each image, it ends up with a .xml file describing all the coins (class, position) in the image.

### Pretrained Model
We used `model = models.resnet50(weights='IMAGENET1K_V2')` as the pretrained model.

### Training
We split the labeled data into training set and validation set (size ratio = 9:1) and train 25 epochs. After each epoch, we calculate the CrossEntropyLoss on the validation set, and preserve the model with the least validation loss.