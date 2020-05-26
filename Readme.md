This project explores using a DCGAN to generate realistic terrain using a dataset created from NASA's opensource Mars DEM data. 

If this experiment is to be replicated you will have to find a large Tiff file to use as input and configure directories within the code. An important note is that this project is configured to generate height maps of 128px only, however increasing this number will lead to a much greater training time. I was able to process 10,000 epochs using a single, top end GPU. If you increase to 1024px, Nvidia recommend 1 week training time with 8 GPUs, or 5 weeks with a single GPU, though this will vary and depend greatly on the number of epochs you chose.

The order of execution is:
TiffSplit
BatchImagePrep
GANTraining
PandasResults

CUDA toolkit needs to be installed and TiffSplit requires the gdal package, which the writer found was easiest accessed using the Anaconda3 environment and could not be accessed in the writer's IDE of choice (PyCharm). The other python files should work in an IDE of your chosing if you install the required packages.