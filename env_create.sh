#!/bin/bash

pip3 install -r requirements.txt
curl -L -o suicide-tweet.zip https://www.kaggle.com/api/v1/datasets/download/duybuingoc/suicide-tweet
unzip suicide-tweet.zip
rm -rf suicide-tweet.zip
python -m spacy download en_core_web_sm