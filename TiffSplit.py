import os, random
from osgeo import gdal


in_path = 'C:/Users/Ezzune/Desktop/'
input_filename = 'Mars.tif'

out_path = 'D:/Map/'
output_filename = 'tile_'

tile_size_x = 512   #Change sizes to fit your requirements. I advise x and y to be equal.
tile_size_y = 512
max_images = 300000

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize


for i in range(0, max_images+1, 1):
    rand_x = random.randint(0, xsize-tile_size_x)
    rand_y = random.randint(0, ysize-tile_size_y)
    com_string = "gdal_translate -of PNG  -ot Byte -scale -srcwin " + str(rand_x)+ ", " + str(rand_y) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + "img_"+ str(i) +str(output_filename) + str(rand_x) + "_" + str(rand_y) + ".png"
    os.system(com_string)
