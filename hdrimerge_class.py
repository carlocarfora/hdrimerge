import sys
import os
import cv2 as cv
import numpy as np
import piexif
import piexif.helper


class Photo:
    ''' Photo class to store individual photo data '''

    def __init__(self, image):
        self.image = image

    def imageCV(self, image):
        ''' Return image as cv image array '''
        im = cv.imread(image)
        return im

    def exifData(self):
        ''' Read exif data from image '''
        exif = piexif.load(self.image)
        exposure_time = exif["Exif"][piexif.ExifIFD.ExposureTime]
        exposure = exposure_time[0] / exposure_time[1]
        return exposure


class PhotoSet:
    ''' PhotoSet class to process a set of Photo objects '''

    def __init__(self):
        self.image_paths = []
        self.im_list = []
        self.exposure_times = []

    def readImages(self, folder):
        """ Return all images in dir and populate im_list with images
        as cv2 data """
        for root, dirs, files in os.walk(folder):
            for file in files:
                p = Photo(file)
                full_path = os.path.join(root, p.image)
                self.image_paths.append(full_path)
                self.im_list.append(p.imageCV(full_path))

    def createHDR(self):
        """ Numpy array of exposure times from exif data """
        for image in self.image_paths:
            p = Photo(image)
            self.exposure_times.append(p.exifData())
        times_np = np.asarray(self.exposure_times, dtype=np.float32)

        # Align Images
        alignMTB = cv.createAlignMTB()
        alignMTB.process(self.im_list, self.im_list)

        # Find Camera Response Curve
        calibrateDebevec = cv.createCalibrateDebevec()
        responseDebevec = calibrateDebevec.process(self.im_list, times_np)

        # Merge Images
        mergeDebevec = cv.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(self.im_list, times_np,
                                          responseDebevec)
        # Generate HDR image and LDR tone mapped preview
        cv.imwrite("hdr.hdr", hdrDebevec)
        toneMapReinhard = cv.createTonemapReinhard(1.5, 0.0)
        ldrReinhard = toneMapReinhard.process(hdrDebevec)
        cv.imwrite("hdr_preview.jpg", ldrReinhard * 255)



folder = "images"
photo = "images/R0010041.JPG"
images = PhotoSet()
images.readImages(folder)
images.createHDR()
hdr = cv.imread('hdr_preview.jpg')
cv.imshow('HDR', hdr)
cv.waitKey(0)
cv.destroyAllWindows()