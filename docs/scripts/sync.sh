#!/bin/bash

# Sync README.md with modified image path for docs/index.md
sed 's|docs/assets/logo.png|assets/logo.png|g' README.md > docs/index.md

# Sync articles
quarto render docs/articles/get-started.qmd
quarto convert docs/articles/get-started.qmd
jupyter nbconvert --to python docs/articles/get-started.ipynb --output ../../examples/get-started.py
rm docs/articles/get-started.ipynb
black examples/get-started.py

quarto render docs/articles/benchmark.qmd
quarto convert docs/articles/benchmark.qmd
jupyter nbconvert --to python docs/articles/benchmark.ipynb --output ../../examples/benchmark.py
rm docs/articles/benchmark.ipynb
black examples/benchmark.py
