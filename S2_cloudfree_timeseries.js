//  -------------------------------------------------------------------
// --------------------  INPUT USERS PARAMETERS  ----------------------
//  -------------------------------------------------------------------

//  1. Import as an asset the AOI you want to use. 
//     Then enter the name of the asset below (roi = ' your AOI')

var roi = table;

//  2. Choose a start date and an end date

var startDate = '2020-01-01';
var endDate = '2021-01-01';

//  3. Choose a 'MGRS_TILE' (Optional)
//     If you don't know which tile to use, just comment out the line below.
//     Then you can run the code once and check in the console tab what tiles there are in the image collection. The Tile name is the last 6 characters of the image name, excluding the 'T' in the front, which stands for 'Tile'.
//     If you enter a tile name and you got error check if the name is correct
var tile = '33UUP';
//33UUP, 32UQU


//  5. Choose the range of images to be desplayed (starts at 0!)

var startImage = 15;
var endImage = 20;

//  6.Explore the images and note down the index 
//  of good images from sorted collection that should be downloaded

var good = [0,1,2,3,4,5,6,7,8,9,11,12,14,15]; //first evaluation
var show_good = 'Y';

//  7. Download images? Y/N

var download = 'N';


//  -------------------------------------------------------------------
// --------------------  MAIN CODE  ----------------------
//  -------------------------------------------------------------------

var s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(roi))
  .select('B2', 'B3', 'B4','B5','B6','B7', 'B8A', 'B8', 'B11','B12')// change bands !!!!!!!!!!!!!!!!!!!!!
  .map(function(image){return image.clip(roi)});
  
print(s2Collection,"S2Collection All Tiles");

if (typeof tile !== 'undefined') {
      // 'tile' is declared, so add the filter and print the collection
     s2Collection = s2Collection.filterMetadata('MGRS_TILE', 'equals', tile);
     print(s2Collection,"S2Collection " + tile);
    }

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
print(sortedCollection, "Sorted collection2");

//----------------------------------------------------------------------
//Display the chosen range of images from the sorted cloud collection
var colstack_list = sortedCollection.toList(sortedCollection.size()); 
print('colstack_list',colstack_list);

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

for(var i = 0; i < good_size; i++)
{
  var index = good[i];
  //print (index);
  var image = ee.Image(colstack_list.get(index));
  print(image);
  var date = image.date().format('yyyy_MM_dd').getInfo();
  if (show_good === 'Y'){
  Map.addLayer(image,vizParams, index.toString()+"_"+ date, 0)
  }
 // Map.addLayer(image,vizParams, index.toString()+"_"+ date, 0)
 print(image,index.toString()+"_"+ date);

}

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
