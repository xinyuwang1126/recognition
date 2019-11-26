#!/usr/bin/env bash

cd /scratch/
#cd /tmp/
#rm -r xinyuw3
#mkdir xinyuw3
#datadir = '/tmp/xinyuw3/'
#cd /tmp
#cp -r /home/xinyuw3/input_file /tmp/xinyuw3/
cd /home/xinyuw3/recognition_code
#mv ../input_file/ /scratch/xinyuw3/
#mv ../input_file/ /tmp/xinyuw3/
#rsync -av --progress '../input_file/' /tmp/xinyuw3/
source activate pytorch0.4
python3 situation_train.py
#rm -rf $datadir
#`mv /scratch/xinyuw3/input_file /home/xinyuw3
