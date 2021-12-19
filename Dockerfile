FROM python:3.8.2

RUN mkdir /home/updater

COPY ./src/requirements.txt /home/updater
RUN pip install -r /home/updater/requirements.txt

COPY ./src/. /home/updater

WORKDIR /home/updater
CMD ["python", "./main.py"]