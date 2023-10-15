# What is this branch
This branch is for personal testing on how to upload a model to jetson infernece and use it.
There seems to be plenty of documentation on how to use custom models. Often times they cover the following:
- pytorch and tensorRT (not live inference - requiring batch size stuff)
- jetsonInference required (don't want to have to use jetosn inference)

# Custom models
What we want would be to ideally have a tensorRT engine running live inference.
This is because we want to move away from jetson inference to models from other papers.
Even though it seems that jetson inference can use these custom models, moving from it allows more flexibility with other aspects

Here are some of the sources I looked at:
tensor RT and pytorch:
https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb

https://medium.com/@heldenkombinat/image-recognition-with-pytorch-on-the-jetson-nano-fd858a5686aa

jetson inference:
https://github.com/dusty-nv/jetson-inference/issues/786#issuecomment-721973736

https://forums.developer.nvidia.com/t/how-to-use-onnx-to-tensorrt-py-with-a-webcam/121568/2

live video:
https://pytorch.org/blog/running-pytorch-models-on-jetson-nano/

# Models I want to use

