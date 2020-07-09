# E3D: Recovering Shape from Events

## Running E3D

#### Installing Pytorch3D
* [Linux - Ubuntu16+]
* [Pytorch3D]
* [Python 3.6+]
* [Pytorch 1.4 or 1.5]
* [gcc & g++ 4.9+]
* [fvcore]
* [CUDA 9.2+ (optional)]

Install dependencies: 
```
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install -c pytorch pytorch torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore fvcore
```
Installing Pytorch3D with CUDA Support (Conda build is currently down - install from source instead):
```
pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
```
Installing Pytorch3D without CUDA Support
```
pip install pytorch3d
```


#### [SUGGESTED] EC2 Instance & Deep Learning AMI
* [Deep Learning Base AMI](https://aws.amazon.com/marketplace/pp/B07Y3VDBNS)
* Instance Type:
    * testing: g4dn.xlarge
    * training: g4dn.4xlarge
* Storage: 
    * 50Gb of ssd

### OPTION 1 - [DOCKER]()
```
docker pull **COMING**
docker run **WAIT**
```

### OPTION 2 - MANUAL INSTALL WITH ANACONDA
#### 1. Install Anaconda for Ubuntu
```
## You can visit (https://www.anaconda.com/distribution/) to install a different version of Anaconda
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

## Check the sum 
sha256sum Anaconda3-2020.02-Linux-x86_64.sh

## Run the script and answer 'yes' to everything
bash Anaconda3-2020.02-Linux-x86_64.sh
```

#### 2. Create and activate the Tensorflow environment
```
source ~/.bashrc
```
Follow instructions above for installing Pytorch3d
    
#### 3. Clone the repo
```
git clone https://github.com/alexisbdr/E3D
```
