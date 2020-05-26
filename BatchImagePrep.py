
import os
import numpy as np
from PIL import Image

ImageSize = 128
ImageChannelsCount = 1
ImagePath = 'D:/Map/' #Change to your image directory
ImageDir = ImagePath
trainingData = []

filelist= [file for file in os.listdir(ImageDir) if file.endswith('.png')] #Select only png files

i = 1


for filename in filelist[:150000]:
    path = os.path.join(ImageDir, filename)
    image = Image.open(path).resize((ImageSize, ImageSize), Image.ANTIALIAS)
    extrema = image.convert("L").getextrema()
    if extrema[1] - extrema[0] >= 25:            #If the image's lowest colour and highest colour are in a close margin
        trainingData.append(np.asarray(image))
    i+=1
    print(i)




trainingData = np.reshape(
    trainingData, (-1, ImageSize, ImageSize, ImageChannelsCount))
trainingData = trainingData / 127.5 - 1


print('saving file...')
np.save('D:/Python/SameColourRemoved2.npy', trainingData)
print(np.shape(trainingData))
