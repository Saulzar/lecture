# How to install torch. 

* Go to http://torch.ch/docs/getting-started.html and follow instructions. These instructions will install torch in your home directory.

# How to run the example

* Have a look at the dataset using `qlua dataset.lua` which should show you a few of the first few images from the test set.
* Create a model with, `th model.lua` which will save a file called `model.t7`
* Have a look at `test.lua` which will load the model from `model.t7` and evaluate the test set on the model.
* Have a look at `train.lua` which will load the model and train the model using the first part of the training set.
  * There is a variable in `train.lua` called `learningRate` which may need adjusting depending if the learning rate is too high (the error is exploding), OR if the model learns too slowly. Adjust this variable by factors of 10.

* Try out the script `test_train.lua` which will run alternating testing and training, again make sure to fix the `learningRate`.
* How is the model performing? There are a couple of other models available, try out `th model_dropout.lua` or `th model_batch_norm.lua` for Dropout training or Batch Normalization. These are forms of regularization specific to deep neural networks. Do these models perform better?


  
