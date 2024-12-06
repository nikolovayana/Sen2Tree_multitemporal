// Description: The idea of this script is that a user can do a fast visual evaluation of all available images
// for a desired Area of interest (aoi) and time-frime. Once the indeces of all 'good' images are writen in the 
// good[] list, then the code prints out a list of all images ids. This list can be then imported in another code as
// an image collection: var imageCollection = ee.ImageCollection(list)
//  -------------------------------------------------------------------
// --------------------  INPUT USERS PARAMETERS  ----------------------
//  -------------------------------------------------------------------


//  1. Import as an asset the AOI you want to use. 
//     Then enter the name of the asset below (roi = ' your AOI')

var roi = table;

//  2. Choose a start date and an end date

var startDate = '2018-01-01';
var endDate = '2021-01-01';
var print_whole_collection = 'N'; // Do you want to print the collection?

//  3. Choose a 'MGRS_TILE' (Optional)
//     If you don't know which tile to use, just comment out the line below.
//     Then you can run the code once and check in the console tab what tiles there are in the image collection. The Tile name is the last 6 characters of the image name, excluding the 'T' in the front, which stands for 'Tile'.
//     If you enter a tile name and you got error check if the name is correct
var tile = '33UUP';
//33UUP, 32UQU
var print_filt_tiles_collection = 'N'; // Do you want to print the new filterd collection?

// 4. Choose one of the two twin satellites from the Sentinel system (Optional)
// Use this if you observe spatial displacement in the images, 
// otherwise use bot twin satellitess and do image co-registration 
// 
var s2a = 'Sentinel-2A';
var s2b = 'Sentinel-2B';
var twinsat = s2a;
var print_filt_twinsat_collection = 'Y';// Do you want to print the new filterd collection?

//  5. Choose the range of images to be desplayed (starts at 0!)

var startImage = 0;
var endImage = 3;

//  6.Explore the images and note down the index 
//  of good images from sorted collection that should be downloaded

var good = [0,1,2,3,4,5,6,7,9,10,11,12,15,17,19, 20, 25, 27, 32, 35]; //first evaluation
var show_good = 'N';

//  7. Download images? Y/N

var download = 'N';


//  -------------------------------------------------------------------
// --------------------  MAIN CODE  ----------------------
//  -------------------------------------------------------------------

// 1.& 2. Create an image collection for the specified AOI and time-perriod
var s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(roi))
  .select('B2', 'B3', 'B4')// change bands as needed if for download !!!!!!!!!!!!!!!!!!!!!
  .map(function(image){return image.clip(roi)});

if (print_whole_collection === 'Y') {
  print("S2Collection All Tiles",s2Collection);
}
// 3. Check if a certain tile is defined and filter the collection only for this tile
if (typeof tile !== 'undefined') {
  // 'tile' is declared, so add the filter and print the collection
  s2Collection = s2Collection.filterMetadata('MGRS_TILE', 'equals', tile);
  if (print_filt_tiles_collection === 'Y') {
    print("S2Collection " + tile, s2Collection);
  }
}

// 4. 
// Filter for images from only Sentinel-2A
if (typeof twinsat !== 'undefined') {
  // 'tile' is declared, so add the filter and print the collection
  s2Collection = s2Collection.filterMetadata('SPACECRAFT_NAME', 'equals', twinsat);
  if (print_filt_twinsat_collection === 'Y') {
    print("S2Collection " + twinsat, s2Collection);
  }
}
    
// Check if an orbit path is defined
// Set Viz parameters
var vizParams = {
  min: 0.0,
  max: 3000,
  gamma: 1.24,
  bands: ['B4', 'B3', 'B2'],
};


//----------------------------------------------------------
// Sort images based on their cloudy percentage
var sortedCollection =  s2Collection.sort('CLOUDY_PIXEL_PERCENTAGE');
//print(sortedCollection, "Sorted collection2");

//----------------------------------------------------------------------
//Display the chosen range of images from the sorted cloud collection
var colstack_list = sortedCollection.toList(sortedCollection.size()); 
//print('colstack_list',colstack_list);

if (show_good === 'N') {
  for(var i = startImage; i < endImage; i++){
  var image = ee.Image(colstack_list.get(i));
  var date = image.date().format('yyyy_MM_dd').getInfo();
  Map.addLayer(image,vizParams, (i).toString()+"_"+ date, 0);
  }
}

//----------------------------------------------------------------------------------------
//Display images defined as "good" from the users list
var good_size = good.length;
var dates_list = [];
var image_id_list = [];

for(var i = 0; i < good_size; i++)
{
  var index = good[i];
  //print (index);
  var image = ee.Image(colstack_list.get(index));
  //var date = image.date().format('yyyy/MM/dd').getInfo();
  var date = image.date().format('MM_dd_yyyy').getInfo();
  dates_list[i] = date;
  //print(index.toString()+"_"+ date);
  var image_id = 'COPERNICUS/S2_SR_HARMONIZED/'+ (image.id().getInfo());
  image_id_list[i] = image_id;
  if (show_good === 'Y'){
  Map.addLayer(image,vizParams, index.toString()+"_"+ date, 0);
  //print(image);
  }
}
print("Good images id List", image_id_list)
print("Good images' dates", dates_list);

// Export "good" images if this option is chosen

function processAnswer (answer){
  if (answer === 'N') {
    // Do nothing or perform some specific actions for 'no'
    return;
  }
  for(var i = 0; i < good_size; i++){
  var index = good[i];
  print (index);
  var image = ee.Image(colstack_list.get(index));
  print(image);
  var date = image.date().format('yyyy_MM_dd').getInfo();
  Export.image.toDrive({
  image: image,
  description:  index.toString()+"_"+ date,
  scale: 10, // Set the desired spatial resolution
  region: roi, // Set the region of interest
  maxPixels:     655761194
});

}
}
processAnswer(download); 
//----------------------------------------------------------------------------------------

// Define visualization parameters
var outlineParams = {
  color: 'red',
  fillColor: '00000000', // Fully transparent fill (hex code with alpha channel)
  width: 2 // Outline width
};

Map.addLayer(roi.style(outlineParams));
Map.centerObject(roi, 12);
