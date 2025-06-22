#!/bin/bash
set -e

conda create -n Regression_cocoa python=3.10 -y
conda run -n Regression_cocoa pip install -r requirements.txt