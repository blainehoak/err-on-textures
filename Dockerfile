FROM --platform=linux/x86_64 pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        unzip \
        zip 

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install torch torchvision numpy torchmetrics pandas scikit-learn python-dotenv

RUN pip3 install git+https://github.com/RobustBench/robustbench.git