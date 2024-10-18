#!/bin/bash

quarto render docs/articles/get-started.qmd
quarto convert docs/articles/get-started.qmd
jupyter nbconvert --to python docs/articles/get-started.ipynb --output ../../examples/get-started.py
rm docs/articles/get-started.ipynb
black examples/get-started.py
