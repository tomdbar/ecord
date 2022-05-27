FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y gcc

# Install pip packages into conda base.
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
#RUN pip install -r /tmp/requirements.txt -f https://data.pyg.org/whl/torch-1.7.1+cpu.html

RUN rm -rf /tmp/*

# working directory
WORKDIR /home/app/ecord

# Finally, pip install ecord.
COPY . .
RUN pip install -e .

#RUN python setup_cy.py build_ext --inplace --force

ENV PYTHONPATH=$PWD:$PYTHONPATH