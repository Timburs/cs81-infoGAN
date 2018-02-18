# InfoGAN for Facial Recognition

We propose using a specific type of Generative Adversarial Networks, InfoGAN, for a specific type of facial recognition.  When provided a dataset of similar faces, the network should be able to draw out specific similar features from that person and classify those based on the amount of Categorical or Continuous variables provided.

## How to Run

### 1) Create a Virtual Environment

SSH into a lab machine that contains a graphics card, go to the scratch directory and create a virtual environment.
```
ssh tnguyen5@avocado.cs.swarthmore.edu
cd /scratch/tnguyen5/
mkdir cs81GAN
cd cs81GAN
virtualenv
```

### 2) Install Requirements
```
pip3 install --user progressbar2
pip3 install --user progress2
pip3 install --user progress
pip3 install --user progressbar
``` 
Now source activate and pip instal inside virtualenv
```
cd cs81GAN/bin/
source activate
pip install progressbar
pip install progress
pip install progressbar2
```
We want to install tensorflow 1.0.0.  Run the following command and copy the specific version you want and paste it after the second command.

```
curl -s https://storage.googleapis.com/tensorflow |xmllint --format - |grep whl
pip install https://storage.googleapis.com/tensorflow/ PASTE HERE
# For Linux, I have the command ready:
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc1-cp35-cp35m-linux_x86_64.whl
```

#### Path Variables
If you get an error like: file not found libcudart.so.8.0, try:
You may have to add these lines to the bottom of your ~/.bashrc file:
`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64`

### 3) Running Code

Now that we have everything, if we don't, just keep pip installing whatever was missing, let's run some code.
Pull the code from our repo and run this:

 ` python3 infogan/__init__.py --infogan `
  
This is very important, because it sets up the code BEFORE training.  We need to run this everytime we start a new session, otherwise we get scaling errors.  Running this also shows if you're missing any packages.

### 3.1) Grabbing a dataset

We have trained the code on various facial datasets.  Find one of your liking.  For our example, we will use the celebA dataset.  Store all the datasets we want inside a new folder caled /datasets/
Some datasets used are: 
* CelebA
* George Bush, Colin Powell, Gerhard Schroeder
* Yale B11/B12
* Cat/Dog Faces

### 3.2) Training

It's recommended to train on a **screen** or **tmux** session so it can keep training will ssh is disconnected.  Training takes awhile!  Running the following code below will allow us to train a dataset with 20 categorical variables and 1 continous variable.  We can change the architecture of discriminator/generator directly from the command line.

```
python3 train.py --dataset celebA/img_align_celeba/ --scale_dataset 64 64 --batch_size 128 --discriminator conv:4:2:64:lrelu,conv:4:2:128:lrelu,conv:4:2:256:lrelu,conv:4:1:256:lrelu,conv:4:1:256:lrelu,fc:1024:lrelu --generator fc:1024,fc:8x8x256,reshape:8:8:256,deconv:4:1:256,deconv:4:2:256,deconv:4:2:128,deconv:4:2:64,deconv:4:1:1:sigmoid --categorical_lambda 1.0 --continuous_lambda 10.0 --categorical_cardinality 20 20 20 --num_continuous 1 --style_size 128 --plot_every 400 --force_grayscale
```

### 3.3) Command Description

Running the above command has the following parameters:
```
python3 train.py 
--dataset datasets/George_W_Bush/ 
--scale_dataset 64 64 
--batch_size 128 
--discriminator conv:4:2:64:lrelu,conv:4:2:128:lrelu,conv:4:2:256:lrelu,conv:4:1:256:lrelu,conv:4:1:256:lrelu,fc:1024:lrelu 
--generator fc:1024,fc:8x8x256,reshape:8:8:256,deconv:4:1:256,deconv:4:2:256,deconv:4:2:128,deconv:4:2:64,deconv:4:1:1:sigmoid 
--categorical_lambda 1.0 
--continuous_lambda 10.0 
--categorical_cardinality 20 20 20 
--num_continuous 1 
--style_size 128 
--plot_every 400 
--force_grayscale
```

*TODO : Explain parameters here.*

### 4) View Results

As the network is training, it creates a log directory.  We can view the results in tensorboard:

First ssh on a new terminal and go to your directory.  Run the following command:

`tensorboard --logdir img_align_celeba_log`

If you are doing this locally, you can ignore the rest and navigate to localhost on whatever port it stated.

The above command should output an IP address and port.  I have
port 6006 and IP addr 130.58.68.74.  Remember it.  Cancel SSH session.
Open a new SSH session with port forwarding :

```
ssh -L 16006:130.58.68.74:6006 tnguyen5@avocado.cs.swarthmore.edu
cd /scratch/tnguyen5/tensorflow-infogan
tensorboard --logdir img_align_celeba_log
localhost:16006
```

### Acknowledgements
Special thanks to Lisa Meeden for guidence throughout the project.
