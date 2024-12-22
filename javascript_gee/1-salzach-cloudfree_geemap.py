import ee 
from ee_plugin import Map

# Description: The idea of this script is that a user can do a fast visual evaluation of all available images
# for a desired Area of interest (aoi) and time-frime. Once the indeces of all 'good' images are writen in the
# good[] list, then the code prints out a list of all images ids. This list can be then imported in another code as
# an image collection: imageCollection = ee.ImageCollection(list)
#  -------------------------------------------------------------------
# --------------------  INPUT USERS PARAMETERS  ----------------------
#  -------------------------------------------------------------------


#  1. Import as an asset the AOI you want to use.
#     Then enter the name of the asset below (roi = ' your AOI')

roi = table

#  2. Choose a start date and an end date

startDate = '2018-01-01'
endDate = '2021-01-01'
print_whole_collection = 'N'; 

#  3. Choose a 'MGRS_TILE' (Optional)
#     If you don't know which tile to use, just comment out the line below.
#     Then you can run the code once and check in the console tab what tiles there are in the image collection. The Tile name is the last 6 characters of the image name, excluding the 'T' in the front, which stands for 'Tile'.
#     If you enter a tile name and you got error check if the name is correct
tile = '33UUP'
#33UUP, 32UQU
print_filt_tiles_collection = 'N'; 

# 4. Choose one of the two twin satellites from the Sentinel system (Optional)
# Use this if you observe spatial displacement in the images,
# otherwise use bot twin satellitess and do image co-registration
#
s2a = 'Sentinel-2A'
s2b = 'Sentinel-2B'
twinsat = s2a
print_filt_twinsat_collection = 'Y';

#  5. Choose the range of images to be desplayed (starts at 0!)

startImage = 0
endImage = 3

#  6.Explore the images and note down the index
#  of good images from sorted collection that should be downloaded

good = [0,1,2,3,4,5,6,7,9,10,11,12,15,17,19, 20, 25, 27, 32, 35]; 
show_good = 'N'

#  7. Download images? Y/N

download = 'N'


#  -------------------------------------------------------------------
# --------------------  MAIN CODE  ----------------------
#  -------------------------------------------------------------------

# 1.& 2. Create an image collection for the specified AOI and time-perriod
s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
.filter(ee.Filter.date(startDate, endDate)) \
.filter(ee.Filter.bounds(roi)) \
.select('B2', 'B3', 'B4')

def func_pnq(image)return image.clip(roi)};: \
.map(function(image){return image.clip(roi)} \
.map(func_pnq)

if (print_whole_collection == 'Y'):
    print("S2Collection All Tiles",s2Collection.getInfo())

# 3. Check if a certain tile is defined and filter the collection only for this tile
if (typeof tile not == 'undefined'):
    # 'tile' is declared, so add the filter and print the collection
    s2Collection = s2Collection.filterMetadata('MGRS_TILE', 'equals', tile)
if (print_filt_tiles_collection == 'Y'):
        print("S2Collection " + tile, s2Collection.getInfo())



# 4.
# Filter for images from only Sentinel-2A
if (typeof twinsat not == 'undefined'):
    # 'tile' is declared, so add the filter and print the collection
    s2Collection = s2Collection.filterMetadata('SPACECRAFT_NAME', 'equals', twinsat)
if (print_filt_twinsat_collection == 'Y'):
        print("S2Collection " + twinsat, s2Collection.getInfo())



# Check if an orbit path is defined
# Set Viz parameters
vizParams = {
    'min': 0.0,
    'max': 3000,
    'gamma': 1.24,
    'bands': ['B4', 'B3', 'B2'],
}


#----------------------------------------------------------
# Sort images based on their cloudy percentage
sortedCollection =  s2Collection.sort('CLOUDY_PIXEL_PERCENTAGE')
#print(sortedCollection, "Sorted collection2".getInfo())

#----------------------------------------------------------------------
#Display the chosen range of images from the sorted cloud collection
colstack_list = sortedCollection.toList(sortedCollection.size())
#print('colstack_list',colstack_list.getInfo())

if (show_good == 'N'):
    fori in range(startImage, endImage, 1):
        image = ee.Image(colstack_list.get(i))
        date = image.date().format('yyyy_MM_dd').getInfo()
        m.addLayer(image,vizParams, (i).toString()+"_"+ date, 0)



#----------------------------------------------------------------------------------------
#Display images defined as "good" from the users list
good_size = good.length
dates_list = []
image_id_list = []

for(i = 0; i < good_size; i++)
{
    index = good[i]
    #print (index)
    image = ee.Image(colstack_list.get(index))
    #date = image.date().format('yyyy/MM/dd').getInfo()
    date = image.date().format('MM_dd_yyyy').getInfo()
    dates_list[i] = date
    #print(index.toString()+"_"+ date.getInfo())
    image_id = 'COPERNICUS/S2_SR_HARMONIZED/'+ (image.id().getInfo())
    image_id_list[i] = image_id
if (show_good == 'Y'):
            m.addLayer(image,vizParams, index.toString()+"_"+ date, 0)
            #print(image.getInfo())

}
print("Good images id List", image_id_list.getInfo())
print("Good images' dates", dates_list.getInfo())

# Export "good" images if this option is chosen

def processAnswer (answer):
if (answer == 'N'):
        # Do nothing or perform some specific actions for 'no'
        return

    fori in range(0, good_size, 1):
        index = good[i]
        print (index)
        image = ee.Image(colstack_list.get(index))
        print(image.getInfo())
        date = image.date().format('yyyy_MM_dd').getInfo()
        Export.image.toDrive(
        image=image,
        description= index.toString()+"_"+ date,
        scale=10, 
        region=roi, 
        maxPixels=    655761194
        )



processAnswer(download)
#----------------------------------------------------------------------------------------

# Define visualization parameters
outlineParams = {
    'color': 'red',
    'fillColor': '00000000', 
    'width': 2 
}

m.addLayer(roi.style(outlineParams))
m.centerObject(roi, 12)
m