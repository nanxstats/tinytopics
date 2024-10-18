#!/bin/bash
Rscript docs/scripts/logo.R
pngquant docs/assets/logo.png --force --output docs/assets/logo.png
