#!/bin/bash

# Generate logo
Rscript docs/scripts/logo.R
if [ -f "Rplots.pdf" ]; then
    rm Rplots.pdf
fi

# Optimize PNG
pngquant docs/assets/logo.png \
    --force \
    --output docs/assets/logo.png

# Pad the logo to get square favicon and resize
magick docs/assets/logo.png \
    -gravity center \
    -background none \
    -extent 640x640 \
    -resize 512x512 \
    docs/assets/favicon.png

# Optimize PNG
pngquant docs/assets/favicon.png \
    --force \
    --output docs/assets/favicon.png
