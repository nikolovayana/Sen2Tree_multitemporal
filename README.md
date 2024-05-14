# README.md

Some parts of this code werre adapted from:   
https://courses.spatialthoughts.com/end-to-end-gee-supplement.html#multi-temporal-composites-for-crop-classification   


## Repository Overview

This project includes a series of scripts designed for processing and analyzing satellite image time series from the Sentinel-2 mission. The workflows involve generating a cloud-free composite of Sentinel-2 images, pre-processing these images to add vegetation indices and normalize data, training a remote sensing classifier, and visualizing the results through a confusion matrix.

## Prerequisites

Before running these scripts, ensure you have access to Google Earth Engine (GEE), as all scripts are implemented to be executed within the GEE code editor. You should have your own Google Earth Engine account with permissions to upload and manipulate assets.

## Execution Flow
The execution of these scripts should follow the order provided:
1. Start with `S2_cloudfree_timeseries.js` to select appropriate images.
2. Use `S2_preprocessing_timeseries.js` to prepare the data for analysis.
3. Run `S2_multitemporal_classification.js` to classify the preprocessed images.
4. Finally, employ `ConfMatrix.py` to visually assess the classification results.

--------------------------------------------------------------------------------------------------------------
### Step 1: `S2_cloudfree_timeseries.js`
- **Function:** Filters and selects cloud-free Sentinel-2 images based on user-defined criteria (AOI, date range, and optional MGRS tile).
- **Output:** A list of image indices deemed cloud-free and suitable for further processing. These images can be directly used in the next step or downloaded for offline use.

### Step 2: `S2_preprocessing_timeseries.js`
- **Input:** Takes the cloud-free images selected from the first step.
- **Function:** Stacks the images into a single composite, adds vegetation indices, and normalizes the data to prepare for machine learning analysis.
- **Output:** Outputs three versions of the image stack: raw, with added vegetation indices, and normalized. These stacked images are then ready to be used for classification.

### Step 3: `S2_multitemporal_classification.js`
- **Input:** Utilizes the preprocessed images (with options to choose among raw, indexed, or normalized stacks).
- **Function:** Trains a classification model using the stacked images and associated training data, performs the classification, and evaluates the model using test data.
- **Output:** The classified image and related metrics such as feature importance and confusion matrix. These outputs are crucial for understanding model performance.

### Step 4: `ConfMatrix.py`
- **Input:** Takes the confusion matrix generated from the classification script.
- **Function:** Provides a visual representation of the confusion matrix to help in assessing the accuracy and performance of the classification model. Generates in adition a table with users and producerrs accuracy for each class.
- **Output:** A visual chart or table that illustrates how well the model has performed in distinguishing between different land cover types.

<img src="https://git.sbg.ac.at/s1098072/i3/-/raw/main/Code/README_images/Screenshot_2024-05-04_132438.png" width="450" height="400" alt="Example Image">

This structured approach ensures a comprehensive analysis, from image selection and preprocessing to classification and evaluation, leveraging the capabilities of Google Earth Engine for environmental remote sensing tasks.
