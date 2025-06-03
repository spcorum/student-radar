# student-radar
 Stanford CS231N (Spring 2025) Student Project - Deep Learning and Radar

# Model
 Use Test.ipynb to train the load the model
 It assume that you are currently at the following base directory: `/content/drive/MyDrive/student-radar/paper/src`
## Training 
```
python models/trainings/train_conditional.py
```
the trained model for each epoch can be found at /checkpoints, e.g. `paper/checkpoints/model-ryyt9way/model-ryyt9way-epoch-1.weights.h5`
## Load the model
```
python generate/save_conditional_generations.py <model name> <epoch number> <generated dataset name>
```
for example, to load the model at `paper/checkpoints/model-ryyt9way/model-ryyt9way-epoch-1.weights.h5` and save it to a new h5 file called `test`, we can run
```
python generate/save_conditional_generations.py ryyt9way 1 test
```
the generated data will be saved to `data/generated/<generated dataset name>.npy`

To get the full generation, including plots and generated_ra data, run
```
python generate/get_generations.py <model name> <epoch number> <generated dataset name>
```
