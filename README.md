# pyqt-torch-cnn-cifar10-gui
PyQt GUI showcase to use basic pytorch CNN model trained CIFAR10 dataset

Model which is used in this script is image classification model.

I used kaggle notebook to train the model, which needs performance.

You can see the model training code in <a href="https://www.kaggle.com/code/yoonjunggyu/pytorch-cnn-cifar10">here</a>.

"cifar_net.pth" is the model made out of it, which is very small and simple.

That means this model is not accurate. This is more like for studying purpose rather than making model with great accuracy, so don't worry about it.

You can see the whole tutorial in <a href="https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html">pytorch official tutorial site</a>.

## How to Install
1. git clone ~
2. python main.py

After running the script, the model will be loaded. You can see the input line at the upper side of the window. Add any image URL from the web, click the run button, and see what happens.

## Preview
![a](https://github.com/yjg30737/pyqt-torch-cnn-cifar10-gui/assets/55078043/7fc2572d-b26a-4939-8700-a102065eac29)

Right!

![b](https://github.com/yjg30737/pyqt-torch-cnn-cifar10-gui/assets/55078043/9c777b66-b6a3-4818-8300-78b08e56e544)

Wrong!

![c](https://github.com/yjg30737/pyqt-torch-cnn-cifar10-gui/assets/55078043/630950f7-6aae-4931-b92c-5f9ff867cad7)

Right!
