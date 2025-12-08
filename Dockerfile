FROM python:3.12

COPY . /src
WORKDIR /src

RUN pip install .
