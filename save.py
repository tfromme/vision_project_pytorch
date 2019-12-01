#!/usr/bin/env python

import sys
import os.path

import cv2 as cv

downscale_factor = 4
test_percent = .2


def generate_images():
    for filename in sys.argv[1:]:
        img = cv.imread(filename)

        dim = (int(img.shape[1] / downscale_factor), int(img.shape[0] / downscale_factor))

        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        filename = os.path.basename(os.path.dirname(filename)) + '/' + os.path.basename(filename)
        yield filename, img


test_mod = int(1 / test_percent)

i = 0
for filename, img in generate_images():
    intermediate_folder = 'Test' if i % test_mod == 0 else 'Train'
    cv.imwrite(f'scaled_pictures/{intermediate_folder}/{filename}', img)
    i += 1
