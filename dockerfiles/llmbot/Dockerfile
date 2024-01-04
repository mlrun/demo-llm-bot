FROM --platform=linux/amd64 mlrun/ml-models:1.4.1

RUN apt-get update && apt-get install libmagic-mgc libmagic1 -y

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt averaged_perceptron_tagger