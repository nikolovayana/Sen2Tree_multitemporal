//------------------------------------------------------------
//AIM: To add vegetation indexes and do normalization. In total 3 multitemporal stacked images will be created: 
//     1 of the S2 collection, 1 of the same collection but eith added vegetation indeces and 1 with a normalized collection including the vegetation indeces
//------------------------------------------------------------
// 1. Rename the bands of the images to the original S2 namings:
// Band renaming dictionary mapping original names to the new names
var bandNames = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10'];
var newBandNames = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B11', 'B12'];

// Function to rename bands of an image
function renameBands(image) {
  return image.select(bandNames, newBandNames);
}

// Applying the renameBands function to each image
var image = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/1_2020_03_19"));
var image2 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/8_2020_03_24"));
var image3 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/7_2020_04_08"));
var image4 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/3_2020_04_23"));
var image5 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/6_2020_05_18"));
var image6 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/4_2020_06_12"));
var image7 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/11_2020_06_27"));
var image8 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/5_2020_08_01"));
var image9 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/0_2020_08_21"));
var image10 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/14_2020_08_26"));
var image11 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/15_2020_09_05"));
var image12 = renameBands(ee.Image("projects/ee-nikolova100yana/assets/I3/2_2020_09_15"));

// Print one of the images to verify the band names
//print('Renamed bands in image:', image);

//#####################################################################################################
//***************** STACKED IMAGE FROM Sentinel_2020_33UUP_AOI_buffer_150m_GeoReference_S2a *****************
//########################################################################################################

// 2. Combine all the images you have imported into an image collection
// Create an image collection from the list of images
var imageCollection = ee.ImageCollection.fromImages([
    image, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12
]);

// Print the image collection to the console
print('Image Collection:', imageCollection);

// Stacking all images in the collection into one multi-band image
var stackedImage = imageCollection.toBands();
print('stackedImage: ', stackedImage);

// Get the list of all band names
var bandNames1 = stackedImage.bandNames();

// Map over each band name and swap the parts around the underscore
var renamedBandNames1 = bandNames1.map(function(name) {
  // Split the name at the underscore, reverse it, and join it back with an underscore
  return ee.String(name).split('_').reverse().join('_');
});

var renamedStackImage = stackedImage.rename(renamedBandNames1);
// print('Original Band Names:', bandNames);
// print('Renamed Band Names:', renamedVegImage.bandNames());
//------------------------------------------------------------------------------
// Export the stacked Vegetation Imaged to the GEE Assets
Export.image.toAsset({
  image: renamedStackImage,
  description: 'stackedImage',
  scale: 10,  // Change this to the appropriate resolution
//  region: stackedImage.geometry(),  // Specify the region, here it defaults to the image's extent
  maxPixels: 1e9,  // Adjust based on the size of your data
});

//#####################################################################################################
//***************** STACKED IMAGE with VEGETATION INDECES *****************
//########################################################################################################

// 3. Add Vegetation indexes: NDVI, EVI, SAVI, NDWI, GCI
// Function to add multiple vegetation indices to an image
var addIndices = function(image) {
  // NDVI: Normalized Difference Vegetation Index
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');

  // EVI: Enhanced Vegetation Index
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename('evi');

  // SAVI: Soil Adjusted Vegetation Index
  var savi = image.expression(
    '(NIR - RED) * (1.5 / (NIR + RED + 0.5))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('savi');

  // NDWI: Normalized Difference Water Index
  var ndwi = image.normalizedDifference(['B8', 'B11']).rename('ndwi');

  // GCI: Green Chlorophyll Index
  var gci = image.expression(
    'NIR / GREEN - 1', {
      'NIR': image.select('B8'),
      'GREEN': image.select('B3')
    }).rename('gci');

  return image.addBands(ndvi).addBands(evi).addBands(savi).addBands(ndwi).addBands(gci);
};

// Apply the addIndices function to each image in the collection
var VegImageCollection = imageCollection.map(addIndices);

// Print the modified image collection to verify that indices have been added
//print('Indexed Image Collection:', VegImageCollection);

//-----------------------------------------------------------------------------
// Create a stack image for the image collection that includes vegetation indeces
// Stacking all images in the collection into one multi-band image
var stackedVegImage = VegImageCollection.toBands();
print('stackedVegImage: ', stackedVegImage);

// Get the list of all band names
var bandNames = stackedVegImage.bandNames();

// Map over each band name and swap the parts around the underscore
var renamedBandNames = bandNames.map(function(name) {
  // Split the name at the underscore, reverse it, and join it back with an underscore
  return ee.String(name).split('_').reverse().join('_');
});

var renamedVegImage = stackedVegImage.rename(renamedBandNames);
// print('Original Band Names:', bandNames);
// print('Renamed Band Names:', renamedVegImage.bandNames());
//------------------------------------------------------------------------------
// Export the stacked Vegetation Imaged to the GEE Assets
Export.image.toAsset({
  image: renamedVegImage,
  description: 'stackedVegImage',
  scale: 10,  // Change this to the appropriate resolution
//  region: stackedImage.geometry(),  // Specify the region, here it defaults to the image's extent
  maxPixels: 1e9,  // Adjust based on the size of your data
});


//#####################################################################################################
//***************** STACKED IMAGE with NORMALIZATION *****************
//########################################################################################################

//------------------------------------------------------------
// 0.2 Normalize all images in an image collection

// Machine learning algorithms work best on images when all features have
// the same range

// Function to Normalize Image
// Pixel Values should be between 0 and 1
// Formula is (x - xmin) / (xmax - xmin)
//************************************************************************** 
function normalize(image){
  var bandNames = image.bandNames();
  // Compute min and max of the image
  var minDict = image.reduceRegion({
    reducer: ee.Reducer.min(),
    geometry: AOI,
    scale: 10,
    maxPixels: 1e9,
    bestEffort: true,
    tileScale: 16
  });
  var maxDict = image.reduceRegion({
    reducer: ee.Reducer.max(),
    geometry: AOI,
    scale: 10,
    maxPixels: 1e9,
    bestEffort: true,
    tileScale: 16
  });
  var mins = ee.Image.constant(minDict.values(bandNames));
  var maxs = ee.Image.constant(maxDict.values(bandNames));

  var normalized = image.subtract(mins).divide(maxs.subtract(mins));
  return normalized;
}

// Apply the normalization to each image in the collection
var NormImageCollection = VegImageCollection.map(normalize);
print(NormImageCollection, "NormImageCollection")

//// Stacking all images in the collection into one multi-band image
var stackedNormImage = NormImageCollection.toBands();
print('stackedNormImage: ', stackedNormImage);

// Get the list of all band names
var bandNamesNorm = stackedNormImage.bandNames();

// Map over each band name and swap the parts around the underscore
var renamedBandNamesNorm = bandNamesNorm.map(function(name) {
  // Split the name at the underscore, reverse it, and join it back with an underscore
  return ee.String(name).split('_').reverse().join('_');
});

var renamedNormImage = stackedNormImage.rename(renamedBandNamesNorm);
print('Original Band Names Norm:', bandNamesNorm);
print('Renamed Band Names Norm:', renamedNormImage.bandNames());
//------------------------------------------------------------------------------
// Export the stacked Vegetation Imaged to the GEE Assets
Export.image.toAsset({
  image: renamedNormImage,
  description: 'stackedNormImage',
  scale: 10,  // Change this to the appropriate resolution
//  region: stackedImage.geometry(),  // Specify the region, here it defaults to the image's extent
  maxPixels: 1e9,  // Adjust based on the size of your data
});
