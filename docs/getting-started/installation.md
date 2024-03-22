For the first time running it will take some time to download the raw dataset and generate the filtered data.

To run the code use linux or WSL, and follow these steps:

<details>
<summary> Important note if having difficulties with TensorFlow!</summary>

There are many problems currently with the Tensorflow package, the easiest way running it with GPU acceleration is in a Colab or Kaggle notebook. Currently a demo notebook is on the way, once the code refactoring is finished. If you are lucky and everything is right you can run it in docker (after installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) or in a venv.

</details>

### Local
Make a virtual environment
```
python3 -m venv env
```
Install requirements

Install tensorflow
```
python3 -m pip install tensorflow[and-cuda]
```
```
pip install -r requirements.txt
```
Run the main script

```
python3 main.py
```

### Docker
1. Clone the repository.
2. Build the dockerfile.
```
sudo docker build -t EEG-ML .
```
1. Run the docker image and mount your working folder
```
docker run --rm -it -p 8888:8888/tcp -v ${PWD}:/workspace EEG-ML:latest
```
1. Run the main script.
```
python3 main.py
```