## LeNet-CNN

A C++ implementation of the popular LeNet convolutional neural network architecture. This example trains on the Kaggle Digit Recognizer dataset - which can be downloaded from [here](https://www.kaggle.com/c/digit-recognizer/data)

### Prerequisites for building and running the model

Dependencies:
- g++ >= 5.0.0
- CMake >= 3.0.0
- make >= 4.0
- Armadillo >= 8.300.4
- Boost unit test framework (Boost version >= 1.58)

This repos was successfully tested in Linux, Ubuntu 18.04

### Building and Running the LeNet on the Digit Recognizer dataset

1. Clone this repository. `https://github.com/Kruthi1907/cnn-LeNet.git`
2. `cd` into the project root (`cd cpp-cnn`) and create the build and data directories using `mkdir build data`.
3. Copy the Kaggle Digit Recognizer dataset into the `data` directory. The `data` directory should now contain two CSV files -- `train.csv` and `test.csv`.
4. `cd` into the build directory (`cd build`) and configure the build using `cmake ../` This will generate a `Makefile` to build the project.
5. Run `make` to build the project. Binaries are written to `build/bin`.
6. Train the model on the Kaggle data using `bin/le_net`.

The program will write the test predictions after each epoch of training into CSV files - `build/results_epoch_1.csv`, `build/results_epoch_2.csv` etc. These files can directly be uploaded to the [submission page](https://www.kaggle.com/c/digit-recognizer/submit) on Kaggle to view the scores.
