import ee 
ee.Initialize()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# This sfunction is included for testing purposes only
def greet():
    print("Hello, world!")

#####################################################################################################
#---------------------------LIST OF DATES---------------------------------------

# Insert list of desired image ids (or create your own collection)
def import_dates_list ():
    dates_list = ["COPERNICUS/S2_SR_HARMONIZED/20200319T101021_20200319T101336_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190921T101031_20190921T101152_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200915T101031_20200915T101550_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20181016T101021_20181016T101021_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20191001T101031_20191001T101659_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20180827T101021_20180827T101023_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20180419T101031_20180419T101457_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20180817T101021_20180817T101024_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200518T101031_20200518T101258_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200408T101021_20200408T101022_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190723T101031_20190723T101347_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190603T101031_20190603T101642_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190613T101031_20190613T101027_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200627T101031_20200627T101243_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200826T101031_20200826T101345_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200905T101031_20200905T101637_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20200508T101031_20200508T101648_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20180807T101021_20180807T101024_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190524T101031_20190524T101701_T33UUP",
    "COPERNICUS/S2_SR_HARMONIZED/20190424T101031_20190424T101032_T33UUP"]
    return dates_list

#####################################################################################################
#---------------------------COLLECTION TO STACK IMAGE FUNTION---------------------------------------

def collectionToStackImage (imageCollection):
    """
    Takes an collection of Sentinel-2 images and converts it into a single multitemporal multiband stacked image. 
    Suitable for multitemporal analyses and creating multitemporal stacked images from Google Earth Engine image collections.

    Args:
        imageCollection (ee.ImageCollection): The collection of images. 
        All bands in the collection's images will be used for creating the final stacked image

    Returns:
        ee.Image: The multitemporal stacked image. Each band of the stacked image is named as (date + '_' + band)
    """
    # Stacking all images in the collection into one multi-band image
    stackedImage = imageCollection.toBands()

    # Get the list of all band names
    bandNames = stackedImage.bandNames()
    # print('Original Band Names:', bandNames.getInfo())

    # 'Current band name': T101021_20200319T101336_T33UUP_B2
    # We want to make it shorter and include only the date (first 8 characters)
    # together with the band name (B2, B3, B4 etc)
    # 'New band name': 20200319_B2
    

    # a function to create new shor band names in the stacked image
    def renameStackImageBands (band_name):
        date = ee.String(band_name).slice(0,8)
        band = ee.String(band_name).slice(39)
        return (date.cat('_').cat(band))


    # Map over each band name and swap the parts around the underscore
    renamedBandNames = bandNames.map(renameStackImageBands)

    # Now replace the old band names in the stack image with the newly created names
    renamedStackImage = stackedImage.rename(renamedBandNames)
    # print('Renamed Band Names:', renamedStackImage.bandNames().getInfo())

    return renamedStackImage



######################################################################################################
#-------------------------ADD VEGETATION INDECES FUNCTION------------------------------------------------

# Add Vegetation indexes: NDVI, Greenness
# Function to add multiple vegetation indices to an image
def addIndices(image):
    # NDVI: Normalized Difference Vegetation Index
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')

    greenness = image.expression(
    'max(0, (NIR / RED) + (NIR / SWIR) - (RED / SWIR))', {
        'NIR': image.select('B8'),   
        'RED': image.select('B4'),  
        'SWIR': image.select('B11') 
    }
    ).rename('greenness')

    return image.addBands(ndvi).addBands(greenness)


###################################################################################
#----------------------------NORMALIZATION FUNCTION----------------------------

# Machine learning algorithms work best on images when all features have
# the same range. Eventhough RF models don't care about normalizartion,
# it will assure easier transferability of the model
#
# Function to normalize all images in an image collection
# Pixel Values should be between 0 and 1
# Formula is (x - xmin) / (xmax - xmin)

def normalize(image):
    bandNames = image.bandNames()
    # Compute min and max of the image
    minDict = image.reduceRegion(
    reducer=ee.Reducer.min(),
    geometry=aoi,
    scale=10,
    maxPixels=1e9,
    bestEffort=True,
    tileScale=16
    )
    maxDict = image.reduceRegion(
    reducer=ee.Reducer.max(),
    geometry=aoi,
    scale=10,
    maxPixels=1e9,
    bestEffort=True,
    tileScale=16
    )
    mins = ee.Image.constant(minDict.values(bandNames))
    maxs = ee.Image.constant(maxDict.values(bandNames))

    normalized = image.subtract(mins).divide(maxs.subtract(mins))
    return normalized

###################################################################################
#----------------------------PLOT HEATMAP----------------------------

def plot_importance(classifier, band_order = None,width=12.8, height=6.8):
    """
    Plots a heatmap from a dictionary, using 'Date id' as y-axis labels and no specific band order.

    Parameters:
    - classifier: ee.Classifier from which importance values will be calculated.
    - band order: a list of band names by which the importance graph will be ordered (optional)
    - width: width of the graph
    - height: height of the graph
    """

    # Run .explain() to see what the classifer looks like
    # print(classifier.explain().getInfo(),"Classifier explain")

    # Calculate variable importance
    importance = ee.Dictionary(classifier.explain().get('importance'))

    # Calculate relative importance (imortance values are normalized into percentages)
    sum = importance.values().reduce(ee.Reducer.sum())

    def func_bid(key, val):
        return (ee.Number(val).multiply(100)).divide(sum)

    relativeImportance = importance.map(func_bid) # variable is of type ee.dictionary.Dictionary

    importance_dict = relativeImportance.getInfo() # create a python dictionary
    
    # Step 1: Convert the dictionary to a DataFrame
    formatted_data = {'Date id': [], 'Band': [], 'Importance': []}

    for key, value in importance_dict.items():
        # Split the key into 'Date id' and 'Band'
        date_id, band = key.split('_')

                # Convert 'YYMMDD' to 'MM/DD/YY' format
        year = date_id[:4]
        month = date_id[4:6]
        day = date_id[6:]
        date_id = f"{month}/{day}/{year}"

        formatted_data['Date id'].append(date_id)
        formatted_data['Band'].append(band)
        formatted_data['Importance'].append(value)

    # Create a DataFrame
    data = pd.DataFrame(formatted_data)

    # Step 2: Pivot the table to create the heatmap-friendly format
    pivot_table = data.pivot_table(index='Date id', columns='Band', values='Importance', aggfunc='mean')[band_order]

    # Step 3: Plot the heatmap
    colors = ["#fefcfd", "#69858f", "#151972", "#85010e"]
    custom_colormap = LinearSegmentedColormap.from_list("custom_sequential", colors, N=256)

    plt.figure(figsize=(width, height))
    ax = plt.gca()

    sns.heatmap(pivot_table, linewidth=0.5, cmap=custom_colormap, fmt=".3f")

    # Customize labels and ticks
    plt.xlabel('Band', fontsize=12, fontweight='bold')
    plt.ylabel('Date', fontsize=13, labelpad=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)  # No custom rotation for dates
    plt.show()

###################################################################################
#----------------------------ACCURACY ASSESMENT----------------------------
def accuracy_assesment (classified_image, testing_data, class_label, order = None ):
    """
    Args:
        classified_image (ee.Image): The image on which classification has been done
        testing_data (): Data points with reference/validation information for the classification.
        class_label (String): the name of the column containing the classes of the classification
        order (List): List of the classes values. 
        This is the order in which classes will appear in the confusion matrix and the returned list of fscore accuracy. Default to None.

    Returns:
        ConfusionMatrix: Nested list representing the confusion matrix
        overallaccuracy: Overall accuracy of the classification
        fscore: Fscore for each of the classes (as a list)
    """

    test = classified_image.sampleRegions(
    collection=testing_data,
    properties= [class_label],
    tileScale=16,
    scale=10,
    )

    eeConfusionMatrix = test.errorMatrix(
    actual=class_label,
    predicted='classification',
    order=order
    )

    ConfusionMatrix = eeConfusionMatrix.getInfo()
    overallaccuracy = eeConfusionMatrix.accuracy().getInfo()
    fscore = eeConfusionMatrix.fscore().getInfo()
    # Producers and consumers accuracies can be added to the function. If this is done, alter the return value accordingly
    # producer = eeConfusionMatrix.producersAccuracy().getInfo()
    # consumer = eeConfusionMatrix.consumersAccuracy().getInfo()

    return ConfusionMatrix, overallaccuracy, fscore

###################################################################################
#----------------------------PLOT CONFUSION MATRIX----------------------------
def plot_confmatrix (confusion_matrix, labels ,overall_accuracy = None):
    # Define the confusion matrix data as a numpy array
    data = np.array(confusion_matrix)

    # Create DataFrame
    cf_frame = pd.DataFrame(data, index=labels, columns=labels)

    sns.heatmap(cf_frame, annot=True, cmap="rocket")

    # Get the current axis
    ax = plt.gca()

    # Set y-axis labels to horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('StackedImage', pad=20)

    # Display accuracies
    #plt.figtext(0.5, 0.05, f"Overall Accuracy: {overall_accuracy:.2f}", ha='center')
    plt.xlabel('Predicted label\n\nOverall accuracy={:0.4f}'.format(overall_accuracy))
    #plt.figtext(0.5, -0.01, f"Cohen's Kappa: {kappa:.2f}", ha='center')

    # Show the plot
    plt.show()

###################################################################################
#----------------------------Train and apply RF model----------------------------

def random_forest_classification (image, training_data, class_label):
    training = image.sampleRegions(
    collection=training_data,
    properties=[class_label],
    scale=10,
    tileScale=16
    )

    #----------------------------------------------------------------------------------
    # 3. Train a classifier.
    #Previous training: numberOfTrees: 200, bagFraction: 0.5 (default)
    # Below are parameters obtaied from HyperParameter Tunning on stackedNormImage_noMarch
    classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=150,
    bagFraction=0.9
    ) \
    .train(
    features=training,
    classProperty=class_label,
    inputProperties=image.bandNames()
    )


    # 4. Apply the classifier.
    classified = image.classify(classifier)

    return classified, classifier


###################################################################################
#----------------------------Create pandas dataframe of a chosen band for Timeseries plot----------------------------
def create_pd_df (single_band_image, training_fc, class_label):
    
    training_data = single_band_image.sampleRegions(
        collection=training_fc,
        properties=[class_label],
        scale=10,
        tileScale=16
        )

    # Convert the FeatureCollection to a list of dictionaries (client-side operation)
    features = training_data.getInfo()['features']

    # Extract properties from each feature
    data = [feature['properties'] for feature in features]

    # Initialize an empty list to store long-format rows
    long_format_data = []

    # Loop through each feature's properties

    for feature in data:
        label = feature[class_label]  # Get the label for this feature
        for band_name, value in feature.items():
            if band_name != class_label:  # Exclude the label column from band data
                # Split the band name into the date and actual band name
                date = band_name.split('_', 1)[0]  # Extract 'yyyymmdd' and 'bandname'
                
                # Format the date
                f_date_int = int(f"{date[4:6]}{date[6:8]}{date[:4]}")  # 'mmddyyyy'
                f_date_str = f"{date[4:6]}/{date[6:8]}/{date[:4]}"  # 'mm/dd/yyyy'
                
                long_format_data.append({
                    'date_int': f_date_int,  # Add formatted date
                    'date_str': f_date_str,
                    'label': label,
                    'values': value
                })


    # Convert the long-format data to a Pandas DataFrame
    df_long = pd.DataFrame(long_format_data)

    df_sorted = df_long.sort_values(by='date_int', ascending=True)

    return df_sorted

###################################################################################
#----------------------------Timeseries plot----------------------------

def plot_timeseries(df,label_map, label_color_map, y_axis_label):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 14  # set width to 10 inches
    fig_size[1] = 7   # set height to 6 inches
    plt.rcParams["figure.figsize"] = fig_size

    # Map your original labels to custom labels
    df['Tree Species'] = df['label'].map(label_map)

    sns.set_theme()
    # sns.set_style("whitegrid")
    graph = sns.lineplot(x='date_str',y='values', hue = 'Tree Species', data=df, palette=label_color_map)
    #sns.set_style("ticks")
    # Rotate the x-axis labels to vertical
    plt.xticks(rotation=70)  # Rotate labels to 90 degrees for vertical alignment
    plt.xlim(left=-1)  # Adjust this value based on your data to add space at the beginning of the x-axis
    # Change the color of the plot area to light gray
    # plt.gca().set_facecolor("#f4f4f4")

    # Add a legend with italic labels
    plt.legend(title='', 
            loc='upper left', 
            bbox_to_anchor=(1, 1), 
            frameon=False, 
            edgecolor='none', 
            prop={'style': 'italic', 'size': 14})  # Set text to italic and adjust the size


    # Optionally, add labels, title, and grid
    plt.xlabel('Dates',fontsize=14)
    plt.ylabel(y_axis_label,fontsize=14)

    print (graph)
# plt.savefig('my_lineplot.png',dpi = 1000)