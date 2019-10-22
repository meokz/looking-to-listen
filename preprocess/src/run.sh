#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "convert_to_jpg_wav.py" \
&& python3 convert_to_jpg_wav.py \
&& echo "convert_to_face.py" \
&& python3 convert_to_face.py \
&& echo "convert_to_vector.py" \
&& python3 convert_to_vector.py \
&& echo "convert_to_stft.py" \
&& python3 convert_to_stft.py
