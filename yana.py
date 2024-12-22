import ee 
import geemap.foliumap as geemap
ee.Initialize()


def greet():
    print("Hello, world!")

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

