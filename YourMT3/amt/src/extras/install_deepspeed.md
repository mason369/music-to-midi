"""

# not required on pytorch 2.0:latest container
pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64

apt-get update
apt-get install git
apt-get install libaio-dev

DS_BUILD_OPS=1 pip install deepspeed
ds_report


pip install deepspeed==0.7.7

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

In case you have trouble building apex from source we recommend using the NGC containers
 from here which come with a pre-built PyTorch and apex release.

nvcr.io/nvidia/pytorch:23.01-py3

pip install deepspeed, pip install transformers[deepspeed]
https://www.deepspeed.ai/docs/config-json/#autotuning

"""
