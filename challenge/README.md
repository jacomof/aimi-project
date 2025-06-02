## ULS23 Baseline Model Container
This folder contains the code to build a GrandChallenge algorithm out of the faster version of the ULS baseline model. This model differs from the original baseline model in two ways:
1. It accepts input images of 64x128x128 instead of the original 128x256x256
2. It is slightly more shallow than the original

- `/architecture/extensions/nnunetv2` contains the extensions to the nnUNetv2 framework that should be merged with your local install.
- `/architecture/input/` contains an example of a stacked VOI image and the accompanying spacings file. Uncommenting line 63 and 66 in the Dockerfile will allow you to run your algorithm locally with this data and check whether it runs inference correctly. 
- `/process.py` is where the model is loaded, predictions are made and postprocessing is applied. If you're testing this model locally and want to use a CPU instead of a GPU, you can do this by changing 'cuda' to 'cpu' in line 23 of process.py.
