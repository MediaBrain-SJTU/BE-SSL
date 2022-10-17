#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author      : changfeng3168
# @time        : 22/10/17 15:15:46
# @description : candi unpack to target dir
# @reference   : 

import os 

pack_list = ['BPDwoPsy', 'BPDwPsy', 'HC', 'SS']


for pack_name in pack_list:
    os.system('tar -zxvf ./data/candi/raw_data/SchizBull_2008_{}_segimgreg_V1.2.tar.gz'.format(pack_name))

for dir in os.listdir('./SchizBull_2008'):
    source_dir = os.path.join('./SchizBull_2008', dir)
    for file in os.listdir(source_dir):
        if file[-4] == '_':
            source_file = os.path.join(source_dir, file)
            os.system('mv -f {} ./data/candi/raw_data/data'.format(source_file))

os.system('rm -rf ./SchizBull_2008')




