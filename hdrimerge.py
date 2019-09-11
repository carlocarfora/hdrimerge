import os
import cv2 as cv
import numpy as np
import piexif
import piexif.helper


def readImages(folder):
    """ Return all images in dir as cv2 image """
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            images.append(full_path)
    return images


def loadImages(images):
    """ Load images as cv image array """
    im_list = []
    for image in images:
        im = cv.imread(image)
        im_list.append(im)
    return im_list


def readExifData(images):
    """ Get exposure time from images and return name exposure time pairs """
    exposure_times = []
    if len(images) is not 0:
        for image in images:
            exif_dict = piexif.load(os.path.abspath(image))
            exposure_time = exif_dict["Exif"][piexif.ExifIFD.ExposureTime]
            top = (exposure_time[0] / exposure_time[0])
            bottom = (exposure_time[1] / exposure_time[0])
            concat = top / bottom
            exposure_times.append(concat)

    return np.asarray(exposure_times, dtype=np.float32)


def alignImages(images):
    alignMTB = cv.createAlignMTB()
    alignMTB.process(images, images)
    return alignMTB


def findCRF(images, exposure_times):
    calibrateDebevec = cv.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, exposure_times)
    return responseDebevec


def mergeImages(images, times, responseDebevec):
    mergeDebevec = cv.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    cv.imwrite("hdr.hdr", hdrDebevec)


image_files = readImages("images")
times = readExifData(image_files)
im_images = loadImages(image_files)
align = alignImages(im_images)
crf = findCRF(im_images, times)
mergeImages(im_images, times, crf)