This project explores using a DCGAN to generate realistic terrain using a dataset created from NASA's opensource Mars DEM data. 

If this experiment is to be replicated you will have to find a large Tiff file to use as input and configure directories within the code. An important note is that this project is configured to generate height maps of 128px only, however increasing this number will lead to a much greater training time. I was able to process 10,000 epochs using a single, top end GPU. If you increase to 1024px, Nvidia recommend 1 week training time with 8 GPUs, or 5 weeks with a single GPU, though this will vary and depend greatly on the number of epochs you chose.

The order of execution is:
TiffSplit
BatchImagePrep
GANTraining
PandasResults

CUDA toolkit needs to be installed and TiffSplit requires the gdal package, which the writer found was easiest accessed using the Anaconda3 environment and could not be accessed in the writer's IDE of choice (PyCharm). The other python files should work in an IDE of your chosing if you install the required packages.

An example of output when applied to terrain in the Unity environment:

Genuine Terrain: 

![real3](https://user-images.githubusercontent.com/28735104/82899779-71e29c80-9f53-11ea-967b-9e78a57001b8.png)

![real15](https://user-images.githubusercontent.com/28735104/82900005-c1c16380-9f53-11ea-9515-bca88eb47666.png)




Generated terrain:

![gen3](https://user-images.githubusercontent.com/28735104/82899690-5081b080-9f53-11ea-85a0-4a1ded93abe8.png)

![gen23](https://user-images.githubusercontent.com/28735104/82899945-ace4d000-9f53-11ea-81a7-10ca71121950.png)


Output issues: You may notice that the generated output has fluctuating terrain, this is due to only 128px output being created due to a lack of access to appropriate resources. This project is fully scalable to the intended 512px/1024px modes that would remove this fluctuation problem and create believable terrain.
