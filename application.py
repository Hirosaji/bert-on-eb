# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

sys.path.append("./bert_script")
from params import BERT_PRAMS
from extract_features import get_futures
from util import is_japanese

import numpy as np
import codecs
import tensorflow as tf
import json


def checkMount():
    with open("/efs/sample_text.txt") as f:
        output = f.read()
    return output


def cos_simularity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def calc_simlarity(total, each):
    sims = [
        cos_simularity(e["layers"][0]["values"], total["layers"][0]["values"])
        for e in each
    ]
    return sims


class convert_to_simlarity:
    def __init__(self):
        self.output = {"body": None}

    def from_texts(self, body):
        self.output["body"] = body

        # run BERT
        self.output["sims"] = self.texts2similarity()

        return self.output

    def texts2similarity(self):
        body = self.output["body"]
        raw_features = get_futures(BERT_PRAMS, body)

        cls_features = []
        for raw_feature in raw_features:
            cls_feature = list(
                filter(lambda layer: layer["token"] == "[CLS]", raw_feature["features"])
            )
            cls_features.append(cls_feature[0])

        simlarities = calc_simlarity(cls_features[0], cls_features[1:])

        return simlarities


# EB looks for an 'application' callable by default.
application = Flask(__name__)
application.config["JSON_AS_ASCII"] = False

# set convert_to_simlarity class
req = convert_to_simlarity()

# enable CORS
CORS(application)

# check post method behavior
application.add_url_rule(
    "/",
    "index",
    (
        lambda: jsonify(
            {
                "type": "check mount file",
                "context": {
                    "message": "Success!",
                },
            }
        )
    ),
    methods=["GET"],
)

# check whether there is mount file
application.add_url_rule(
    "/mount_check",
    "mount check",
    (
        lambda: jsonify(
            {
                "type": "check mount file",
                "context": {
                    "mount_file": checkMount(),
                },
            }
        )
    ),
    methods=["GET"],
)

# compute sentence similarity
application.add_url_rule(
    "/sim",
    "similarity",
    (
        lambda: jsonify(
            {
                "type": "sentence similarity",
                "context": req.from_texts(request.get_json()["texts"]),
            }
        )
    ),
    methods=["POST"],
)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
