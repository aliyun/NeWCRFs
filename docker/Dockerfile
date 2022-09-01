# Include the base image for the docker
# You can use an image based on PyTorch, Tensorflow, MXNet etc, depending on your prefered machine learning tool.
FROM python:3.8
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime


RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN conda init bash && source ~/.bashrc && echo "conda activate base" >> ~/.bashrc


# Setting working directory to /opt, rather than doing all the work in root.
# Copying the /code directory into /opt
WORKDIR /opt
COPY ./src /opt

# # Running pip install to download required packages
RUN conda env create -f environment.yml


# Setting the default code to run when a container is launced with this image.
ENTRYPOINT [ "/bin/bash", "/opt/run.sh" ]
