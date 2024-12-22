import ee 
from ee_plugin import Map

# 0. Import AOI, stacked multitemporal image, training and testing data.

AOI = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/AOI_Salzachauen_buffer_150m_WGS84_33N")
test_P = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/Test_WGS33N_P")
train_P = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/Train_WGS33N_P")
stackedVegImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedVegImage")
stackedImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedImage")
stackedNormImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedNormImage")
pca = ee.Image("projects/ee-nikolova100yana/assets/I3/pca_StackedNormImage")
stackedNormImage_noMarch = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedNormImage_noMarch")

# 1. Choose a composite (stacked multitemporal image)
#    on which classification to be done

#composite = stackedVegImage
#composite = stackedNormImage
#composite = stackedImage
#composite = pca
composite = stackedNormImage_noMarch
print(composite.getInfo())


#---------------------------------------------------------------------------------

# 2. Overlay the point on the image to get training data.

# OPTIONAL Filtering training data to include only desired classes
filteredTraining = train_P.filter(
ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 8])
)

#train_col = train_P#all species
train_col = filteredTraining
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
training = composite.sampleRegions(
#collection=train_P,
collection=train_col,
properties=['acronym_N'],
scale=10,
tileScale=16
)
print('Training:',training.getInfo())
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
classProperty='acronym_N',
inputProperties=composite.bandNames()
)


# 4. Apply the classifier.
classified = composite.classify(classifier)
print (classified, "classified")

palette = [
'4682b4', 
'32cd32', 
'800080', 
'ffa500', 
'ff6347', 
'40e0d0', 
#'ee82ee', # Violet
#'f5deb3'  # Wheat
]

m.addLayer(classified, {'min': 0, 'max': 7, 'palette': palette}, '2020')
#----------------------------------------------------------------------------
# 5. Calsculate Feature Importance

# Run .explain() to see what the classifer looks like
print(classifier.explain(),"Classifier explain".getInfo())

# Calculate variable importance
importance = ee.Dictionary(classifier.explain().get('importance'))


# Calculate relative importance
sum = importance.values().reduce(ee.Reducer.sum())


def func_bid(key, val):
    return (ee.Number(val).multiply(100)).divide(sum)

relativeImportance = importance.map(func_bid)


print(relativeImportance.getInfo())

# Create a FeatureCollection so we can chart it
importanceFc = ee.FeatureCollection([
ee.Feature(None, relativeImportance)
])

chart = ui.Chart.feature.byProperty(
features=importanceFc
).setOptions(
title='Feature Importance',
vAxis={title='Importance'},
hAxis={title='Feature'},
legend={position='none'}
)
print(chart.getInfo())

#----------------------------------------------------------------------------------
# 6. PCA Analyses

# Define the geometry and scale parameters
geometry = AOI.geometry()
scale = 10

# Run the PCA function
pca = PCA(composite)

# Extract the properties of the pca image
variance = pca.toDictionary()
print('Variance of Principal Components', variance.getInfo())

# As you see from the printed results, ~97% of the variance
# from the original image is captured in the first 3 principal components
# We select those and discard others
pca = PCA(composite).select(['pc1', 'pc2', 'pc3'])
print('First 3 PCA Bands', pca.getInfo())

# PCA computation is expensive and can time out when displaying on the map
# # Export the results and import them back
# geemap.ee_export_image_to_asset(
#   image=pca,
#   description='Principal_Components_Image',
#   assetId='users/ujavalgandhi/e2e/arkavathy_pca',
#   region=geometry,
#   scale=scale,
#   maxPixels=1e10})

# Once the export finishes, import the asset and display
pcaImported = ee.Image("projects/ee-nikolova100yana/assets/I3/pca_StackedNormImage")
pcaVisParams = {bands=['pc1', 'pc2', 'pc3'], min=-2, max=2}

m.addLayer(pcaImported, pcaVisParams, 'Principal Components')
#----------------------------------------------------------------------------------
# 5. Accuracy Assessment
# Use classification map to assess accuracy using the validation fraction
# of the overall training set created above.


# OPTIONAL Filtering training data to include only desired classes
filteredTest = test_P.filter(
#ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 5, 8])# without AcPs and AlGl
ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 8])
)

#test_col = test_P; # all species
test_col = filteredTest

test = classified.sampleRegions(
collection=test_col,
properties=['acronym_N'],
tileScale=16,
scale=10,
)

print('Test =', test.getInfo())
testConfusionMatrix = test.errorMatrix(
actual='acronym_N',
predicted='classification',
order=[1, 2, 3, 4, 8]
)

# Printing of confusion matrix and accuracy
print('Confusion Matrix', testConfusionMatrix.getInfo())
print('Test Accuracy', testConfusionMatrix.accuracy().getInfo())
#----------------------------------------------------

#Print confusion matrix as a text to import in Jupyter Notebook

matrixArray = testConfusionMatrix.array()

# Convert the matrix to a nested list
matrixList = matrixArray.toList()

# This will print the confusion matrix as a list of lists in the console

def func_ajp(result) =
    console.log(JSON.stringify(result))

matrixList.evaluate(func_ajp)




#-------------------------------
# Exporting the confusion matrix to Google Drive as CSV
geemap.ee_export_vector_to_drive(
collection=ee.FeatureCollection([
ee.Feature(None, {matrix=testConfusionMatrix.array()})
]),
description='ConfusionMatrix',
folder='EarthEngine',
fileNamePrefix='confusion_matrix',
#fileFormat=''
)


#----------------------------------------------------------------------------
# 6. Generate a chart for the confusion matrix with class labels
#Adjusted labels array to include a label for class '0'
#labels = ['PiAb', 'PoBa', 'FrEx', 'AlIn', 'QuRo', 'AcPs', 'AlGl', 'SaAl'];# for all species
labels = ['PiAb', 'PoBa', 'FrEx', 'AlIn', 'SaAl']

chart = ui.Chart.array.values(
array=testConfusionMatrix.array(),
axis=0,
xLabels=labels
).setChartType('Table') \
.setOptions(
title='Confusion Matrix',
hAxis={title='Predicted Class', slantedText=True, slantedTextAngle=45},
vAxis={title='Actual Class'},
width='500',
height='300'
)

# # Print the chart to the Console
print(chart.getInfo())

#**************************************************************************
# Function to calculate Principal Components
# Code adapted from https =#developers.google.com/earth-engine/guides/arrays_eigen_analysis
#**************************************************************************
def PCA(maskedImage) =
    image = maskedImage.unmask()
    scale = scale
    region = geometry
    bandNames = image.bandNames()
    # Mean center the data to enable a faster covariance reducer
    # and an SD stretch of the principal components.
    meanDict = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=region,
    scale=scale,
    maxPixels=1e13,
    tileScale=16
    )
    means = ee.Image.constant(meanDict.values(bandNames))
    centered = image.subtract(means)
    # This helper function returns a list of new band names.

    def func_mwb(prefix) =
        seq = ee.List.sequence(1, bandNames.length())

        def func_win(b) =
                return ee.String(prefix).cat(ee.Number(b).int())

        return seq.map(func_win)



    getNewBandNames = func_mwb









    def func_elk(centered, scale, region) =
        # Collapse the bands of the image into a 1D array per pixel.
        arrays = centered.toArray()

        # Compute the covariance of the bands within the region.
        covar = arrays.reduceRegion(
            reducer=ee.Reducer.centeredCovariance(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
            tileScale=16
            )

        # Get the 'array' covariance result and cast to an array.
        # This represents the band-to-band covariance within the region.
        covarArray = ee.Array(covar.get('array'))

        # Perform an eigen analysis and slice apart the values and vectors.
        eigens = covarArray.eigen()

        # This is a P-length vector of Eigenvalues.
        eigenValues = eigens.slice(1, 0, 1)

        # Compute Percentage Variance of each component
        # This will allow us to decide how many components capture
        # most of the variance in the input
        eigenValuesList = eigenValues.toList().flatten()
        total = eigenValuesList.reduce(ee.Reducer.sum())


        def func_zpv(item) =
                component = eigenValuesList.indexOf(item).add(1).format('%02d')
                variance = ee.Number(item).divide(total).multiply(100).format('%.2f')
                return ee.List([component, variance])

        percentageVariance = eigenValuesList.map(func_zpv)




        # Create a dictionary that will be used to set properties on final image
        varianceDict = ee.Dictionary(percentageVariance.flatten())
        # This is a PxP matrix with eigenvectors in rows.
        eigenVectors = eigens.slice(1, 1)
        # Convert the array image to 2D arrays for matrix computations.
        arrayImage = arrays.toArray(1)

        # Left multiply the image array by the matrix of eigenvectors.
        principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage)

        # Turn the square roots of the Eigenvalues into a P-band image.
        # Call abs() to turn negative eigenvalues to positive before
        # taking the square root
        sdImage = ee.Image(eigenValues.abs().sqrt()) \
        .arrayProject([0]).arrayFlatten([getNewBandNames('sd')])

        # Turn the PCs into a P-band image, normalized by SD.
        return principalComponents \
        .arrayProject([0]) \
        .arrayFlatten([getNewBandNames('pc')]) \
        .divide(sdImage) \
        .set(varianceDict)

m