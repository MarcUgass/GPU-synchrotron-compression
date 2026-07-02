#!/bin/bash

INPUT_FILE="Resources/scan01_000003-u16be-1x2527x2463.raw"
OUTPUT_FILE="output/test.enc"
LUT_DIR="LUT/n1_lossless/"
EXECUTABLE="./build/PICSONG"

echo "Midiendo tiempo de ejecución..."

start=$(date +%s.%N)

$EXECUTABLE -wl 5 -cp 2 -type 0 -qs 1 -i "$INPUT_FILE" -o "$OUTPUT_FILE" -cbWidth 64 -cbHeight 18 -cd 0 -xSize 2048 -ySize 2048 -video 0 -isRGB 1 -LUTFolder "$LUT_DIR" -k 0

end=$(date +%s.%N)

duration=$(echo "$end - $start" | bc)

min=$(echo "scale=0; $duration / 60" | bc)
sec=$(echo "scale=0; $duration % 60 / 1" | bc)
ms=$(echo "scale=0; ($duration * 1000 / 1) % 1000" | bc)

echo "----------------------------------"
printf "Duración: %02d:%02d.%03d (MM:SS.mmm)\n" $min $sec $ms