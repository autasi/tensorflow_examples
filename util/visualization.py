#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv

def load_palette(path, skip_first=False, scale01=True):
    palette = []
    with open(path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            palette.append([int(val) for val in row][::-1])
    if skip_first:
        palette = palette[1:]
    if scale01:
        palette = [(p[0]/255.0, p[1]/255.0, p[2]/255.0) for p in palette]
    return palette
