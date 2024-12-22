import ee 
from ee_plugin import Map

#####################################################################################
#----------------------------IMPORTS-------------------------------------------------

# Import area of interest (AOI)
aoi = ee.FeatureCollection("projects/ee-nikolova100yana/assets/AOI_Salzachauen_buffer_150m_WGS84_33N")

# Insert list of desired image ids (or create your own collection)
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

########################################################################################

main()

#---------------------------MAIN FUNCTION-----------------------------------------------

def main():

    # Combine all the images you have imported into an image collection
    imageCollection = ee.ImageCollection(dates_list) \
    .select('B2', 'B3', 'B4','B5','B6','B7', 'B8A', 'B8', 'B11','B12')

    def func_vwi(image)return image.clip(aoi)};: \
    .map(function(image){return image.clip(aoi)} \
    .map(func_vwi)

    print("Image collection", imageCollection.getInfo())

    # Add vegetation indices bands to each image in the collection.
    VegImageCollection = imageCollection.map(addIndices)

    ## Print the modified image collection to verify that indices have been added
    print("VegImageCOllection:", VegImageCollection.getInfo())

    # Apply the normalization to each image in the collection
    NormImageCollection = VegImageCollection.map(normalize)
    print("NormImageCollection", NormImageCollection.getInfo())

    # Create a stack image for the image collection that includes vegetation indeces
    # Stacking all images in the collection into one multi-band image
    stackedImage = collectionToStackImage(VegImageCollection)
    print("stackedImage: ", stackedImage.getInfo())


    # TO DO
    Multitemporal classification on four seasonal composites
    choose seasonal period depending on line graphs results for NDVI and Greenness

    'Grabska uses':
    'early spring': 04/20 - 05/10
    'late spring': 05/15 - 06/05
    'summer': 06/10 - 07/10
    'autumn': 10/25 - 11/10

    #


    # TO DO
    Spectral-Temporal Metrics for the whole year
    -varaition
    -standard deviation
    -mean
    -q1
    -q3

    #


#
1. function collectionToStackImage (imageCollection)
#
#####################################################################################################
#---------------------------COLLECTION TO STACK IMAGE FUNTION---------------------------------------

def collectionToStackImage (imageCollection):
    # Stacking all images in the collection into one multi-band image
    stackedImage = imageCollection.toBands()

    # Get the list of all band names
    bandNames = stackedImage.bandNames()
    print('Original Band Names:', bandNames.getInfo())

    #
    'Current band name': T101021_20200319T101336_T33UUP_B2
    We want to make it shorter and include only the date (first 8 characters)
    together with the band name (B2, B3, B4 etc)
    'New band name': 20200319_B2
    #

    # a function to create new shor band names in the stacked image
    def renameStackImageBands (band_name):
        date = ee.String(band_name).slice(0,8)
        band = ee.String(band_name).slice(39)
        return (date.cat('_').cat(band))


    # Map over each band name and swap the parts around the underscore
    renamedBandNames = bandNames.map(renameStackImageBands)

    # Now replace the old band names in the stack image with the newly created names
    renamedStackImage = stackedImage.rename(renamedBandNames)
    print('Renamed Band Names:', renamedStackImage.bandNames().getInfo())

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
the same range. Eventhough RF models don't care about normalizartion,
it will assure easier transferability of the model
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




m