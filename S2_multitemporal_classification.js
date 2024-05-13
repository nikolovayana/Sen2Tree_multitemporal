// 0. Import AOI, stacked multitemporal image, training and testing data. 

var AOI = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/AOI_Salzachauen_buffer_150m_WGS84_33N"),
    test_P = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/Test_WGS33N_P"),
    train_P = ee.FeatureCollection("projects/ee-nikolova100yana/assets/I3/Train_WGS33N_P"),
    stackedVegImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedVegImage"),
    stackedImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedImage"),
    stackedNormImage = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedNormImage"),
    pca = ee.Image("projects/ee-nikolova100yana/assets/I3/pca_StackedNormImage"),
    stackedNormImage_noMarch = ee.Image("projects/ee-nikolova100yana/assets/I3/stackedNormImage_noMarch");
    
// 1. Choose a composite (stacked multitemporal image) 
//    on which classification to be done

//var composite = stackedVegImage; 
//var composite = stackedNormImage;
//var composite = stackedImage;
//var composite = pca;
var composite = stackedNormImage_noMarch;
print(composite);


//---------------------------------------------------------------------------------

// 2. Overlay the point on the image to get training data.

// OPTIONAL Filtering training data to include only desired classes
var filteredTraining = train_P.filter(
  ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 8])// without AcPs and AlGl
);

//var train_col = train_P//all species
var train_col = filteredTraining;
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
var training = composite.sampleRegions({
  //collection: train_P,
  collection: train_col,
  properties: ['acronym_N'],
  scale: 10,
  tileScale: 16
});
print('Training:',training);
//----------------------------------------------------------------------------------
// 3. Train a classifier.
//Previous training: numberOfTrees: 200, bagFraction: 0.5 (default)
// Below are parameters obtaied from HyperParameter Tunning on stackedNormImage_noMarch
var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 150,
    bagFraction: 0.9 
  })
.train({
  features: training,  
  classProperty: 'acronym_N',
  inputProperties: composite.bandNames()
});


// 4. Apply the classifier.
var classified = composite.classify(classifier);
print (classified, "classified")

var palette = [
  '4682b4', // Steel blue
  '32cd32', // Lime green
  '800080', // Purple
  'ffa500', // Orange
  'ff6347', // Tomato red
  '40e0d0', // Turquoise
  //'ee82ee', // Violet
  //'f5deb3'  // Wheat
];

Map.addLayer(classified, {min: 0, max: 7, palette: palette}, '2020');
//----------------------------------------------------------------------------
// 5. Calsculate Feature Importance

// Run .explain() to see what the classifer looks like
print(classifier.explain(),"Classifier explain")

// Calculate variable importance
var importance = ee.Dictionary(classifier.explain().get('importance'))


// Calculate relative importance
var sum = importance.values().reduce(ee.Reducer.sum())

var relativeImportance = importance.map(function(key, val) {
  return (ee.Number(val).multiply(100)).divide(sum)
  })
print(relativeImportance)

// Create a FeatureCollection so we can chart it
var importanceFc = ee.FeatureCollection([
  ee.Feature(null, relativeImportance)
])

var chart = ui.Chart.feature.byProperty({
  features: importanceFc
}).setOptions({
      title: 'Feature Importance',
      vAxis: {title: 'Importance'},
      hAxis: {title: 'Feature'},
      legend: {position: 'none'}
  })
print(chart)

//----------------------------------------------------------------------------------
// 6. PCA Analyses

// Define the geometry and scale parameters
var geometry = AOI.geometry();
var scale = 10;

// Run the PCA function
var pca = PCA(composite)

// Extract the properties of the pca image
var variance = pca.toDictionary()
print('Variance of Principal Components', variance)

// As you see from the printed results, ~97% of the variance
// from the original image is captured in the first 3 principal components
// We select those and discard others
var pca = PCA(composite).select(['pc1', 'pc2', 'pc3'])
print('First 3 PCA Bands', pca);

// PCA computation is expensive and can time out when displaying on the map
// // Export the results and import them back
// Export.image.toAsset({
//   image: pca,
//   description: 'Principal_Components_Image',
//   assetId: 'users/ujavalgandhi/e2e/arkavathy_pca',
//   region: geometry,
//   scale: scale,
//   maxPixels: 1e10})

// Once the export finishes, import the asset and display
var pcaImported = ee.Image("projects/ee-nikolova100yana/assets/I3/pca_StackedNormImage");
var pcaVisParams = {bands: ['pc1', 'pc2', 'pc3'], min: -2, max: 2};

Map.addLayer(pcaImported, pcaVisParams, 'Principal Components');
//----------------------------------------------------------------------------------
// 5. Accuracy Assessment
// Use classification map to assess accuracy using the validation fraction
// of the overall training set created above.


// OPTIONAL Filtering training data to include only desired classes
var filteredTest = test_P.filter(
  //ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 5, 8])// without AcPs and AlGl
  ee.Filter.inList('acronym_N', [0, 1, 2, 3, 4, 8])// without AcPs and QuRo
);

//var test_col = test_P; // all species
var test_col = filteredTest;

var test = classified.sampleRegions({
  collection: test_col,
  properties: ['acronym_N'],
  tileScale: 16,
  scale: 10,
});

print('Test:', test)
var testConfusionMatrix = test.errorMatrix({
  actual: 'acronym_N',
  predicted: 'classification',
  order: [1, 2, 3, 4, 8]
});

// Printing of confusion matrix and accuracy
print('Confusion Matrix', testConfusionMatrix);
print('Test Accuracy', testConfusionMatrix.accuracy());
//----------------------------------------------------

//Print confusion matrix as a text to import in Jupyter Notebook

var matrixArray = testConfusionMatrix.array();

// Convert the matrix to a nested list
var matrixList = matrixArray.toList();

// This will print the confusion matrix as a list of lists in the console
matrixList.evaluate(function(result) {
  console.log(JSON.stringify(result));
});


//-------------------------------
// Exporting the confusion matrix to Google Drive as CSV
Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(null, {matrix: testConfusionMatrix.array()})
  ]),
  description: 'ConfusionMatrix',
  folder: 'EarthEngine',
  fileNamePrefix: 'confusion_matrix',
  //fileFormat: ''
});


//----------------------------------------------------------------------------
// 6. Generate a chart for the confusion matrix with class labels
//Adjusted labels array to include a label for class '0'
//var labels = ['PiAb', 'PoBa', 'FrEx', 'AlIn', 'QuRo', 'AcPs', 'AlGl', 'SaAl'];// for all species
var labels = ['PiAb', 'PoBa', 'FrEx', 'AlIn', 'SaAl'];

var chart = ui.Chart.array.values({
  array: testConfusionMatrix.array(),
  axis: 0,
  xLabels: labels
}).setChartType('Table')
  .setOptions({
    title: 'Confusion Matrix',
    hAxis: {title: 'Predicted Class', slantedText: true, slantedTextAngle: 45},
    vAxis: {title: 'Actual Class'},
    width: '500',
    height: '300'
});

// // Print the chart to the Console
 print(chart);

//************************************************************************** 
// Function to calculate Principal Components
// Code adapted from https://developers.google.com/earth-engine/guides/arrays_eigen_analysis
//************************************************************************** 
function PCA(maskedImage){
  var image = maskedImage.unmask()
  var scale = scale;
  var region = geometry;
  var bandNames = image.bandNames();
  // Mean center the data to enable a faster covariance reducer
  // and an SD stretch of the principal components.
  var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: scale,
    maxPixels: 1e13,
    tileScale: 16
  });
  var means = ee.Image.constant(meanDict.values(bandNames));
  var centered = image.subtract(means);
  // This helper function returns a list of new band names.
  var getNewBandNames = function(prefix) {
    var seq = ee.List.sequence(1, bandNames.length());
    return seq.map(function(b) {
      return ee.String(prefix).cat(ee.Number(b).int());
    });
  };
  // This function accepts mean centered imagery, a scale and
  // a region in which to perform the analysis.  It returns the
  // Principal Components (PC) in the region as a new image.
  var getPrincipalComponents = function(centered, scale, region) {
    // Collapse the bands of the image into a 1D array per pixel.
    var arrays = centered.toArray();
    
    // Compute the covariance of the bands within the region.
    var covar = arrays.reduceRegion({
      reducer: ee.Reducer.centeredCovariance(),
      geometry: region,
      scale: scale,
      maxPixels: 1e13,
      tileScale: 16
    });

    // Get the 'array' covariance result and cast to an array.
    // This represents the band-to-band covariance within the region.
    var covarArray = ee.Array(covar.get('array'));

    // Perform an eigen analysis and slice apart the values and vectors.
    var eigens = covarArray.eigen();

    // This is a P-length vector of Eigenvalues.
    var eigenValues = eigens.slice(1, 0, 1);
    
    // Compute Percentage Variance of each component
    // This will allow us to decide how many components capture
    // most of the variance in the input
    var eigenValuesList = eigenValues.toList().flatten()
    var total = eigenValuesList.reduce(ee.Reducer.sum())

    var percentageVariance = eigenValuesList.map(function(item) {
      var component = eigenValuesList.indexOf(item).add(1).format('%02d')
      var variance = ee.Number(item).divide(total).multiply(100).format('%.2f')
      return ee.List([component, variance])
    })
    // Create a dictionary that will be used to set properties on final image
    var varianceDict = ee.Dictionary(percentageVariance.flatten())
    // This is a PxP matrix with eigenvectors in rows.
    var eigenVectors = eigens.slice(1, 1);
    // Convert the array image to 2D arrays for matrix computations.
    var arrayImage = arrays.toArray(1);

    // Left multiply the image array by the matrix of eigenvectors.
    var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);

    // Turn the square roots of the Eigenvalues into a P-band image.
    // Call abs() to turn negative eigenvalues to positive before
    // taking the square root
    var sdImage = ee.Image(eigenValues.abs().sqrt())
      .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);

    // Turn the PCs into a P-band image, normalized by SD.
    return principalComponents
      // Throw out an an unneeded dimension, [[]] -> [].
      .arrayProject([0])
      // Make the one band array image a multi-band image, [] -> image.
      .arrayFlatten([getNewBandNames('pc')])
      // Normalize the PCs by their SDs.
      .divide(sdImage)
      .set(varianceDict);
  };
  var pcImage = getPrincipalComponents(centered, scale, region);
  return pcImage.mask(maskedImage.mask());
}

