//-----------------------------------------------------------------------------
//Project Info
// Done by Yana Nikolova
// Code reused from:
// https://courses.spatialthoughts.com/end-to-end-gee-supplement.html#introduction
//----------------------------------------------------------------------------

// 1. Create a stacked image with composites from all seasons
// This multi-temporal image is able capture the tree species phenology

var composite = img1
  .addBands(img8)
  .addBands(img7)
  .addBands(img3)
  .addBands(img6)
  .addBands(img4)
  .addBands(img11)
  .addBands(img5)
  .addBands(img0)
  .addBands(img14)
  .addBands(img15)
  .addBands(img2);
  
//print(composite)

// This is a 120-band image
// Use this image for sampling training points for
// to train a tree species classifier 
//---------------------------------------------------------------------------------

// 2. Overlay the point on the image to get training data.
var training = composite.sampleRegions({
  collection: train_P,
  properties: ['acronym_N'],
  scale: 10,
  tileScale: 16
});
print(training);
//----------------------------------------------------------------------------------
// 3. Train a classifier.
var classifier = ee.Classifier.smileRandomForest(200)
.train({
  features: training,  
  classProperty: 'acronym_N',
  inputProperties: composite.bandNames()
});


// 4. Apply the classifier.
var classified = composite.classify(classifier);

var palette = [
  '1f77b4', // Muted blue
  'ff7f0e', // Safety orange
  '2ca02c', // Cooked asparagus green
  'd62728', // Brick red
  '9467bd', // Muted purple
  '8c564b', // Chestnut brown
  'e377c2', // Raspberry yogurt pink
  '7f7f7f'  // Middle gray
];

Map.addLayer(classified, {min: 0, max: 7, palette: palette}, '2020');
//----------------------------------------------------------------------------

// 5. Accuracy Assessment
// Use classification map to assess accuracy using the validation fraction
// of the overall training set created above.
var test = classified.sampleRegions({
  collection: test_P,
  properties: ['acronym_N'],
  tileScale: 16,
  scale: 10,
});

var testConfusionMatrix = test.errorMatrix('acronym_N', 'classification')
// Printing of confusion matrix may time out. Alternatively, you can export it as CSV
print('Confusion Matrix', testConfusionMatrix);
print('Test Accuracy', testConfusionMatrix.accuracy());
