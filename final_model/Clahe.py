import sys
import os
import os.path
import cv2
import numpy as np

#Inspired from: https://github.com/MasazI/clahe_python_opencv/blob/master/core.py

def clahe_gridsize(image_path, denoise=False, verbose=False, cliplimit=None, gridsize=8):
    """This function applies CLAHE to normal RGB images and outputs them.
    The image is first converted to LAB format and then CLAHE is applied only to the L channel.
    Inputs:
      image_path: Absolute path to the image file.
      denoise: Toggle to denoise the image or not. Denoising is done after applying CLAHE.
      verbose: Toggle to view the preprocessed image in an OpenCV window. Change code to enable it.
      cliplimit: The pixel (high contrast) limit applied to CLAHE processing. Read more here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
      gridsize: Grid/block size the image is divided into for histogram equalization.
    Returns:
      bgr: The CLAHE applied image.
    """
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        #bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr

if __name__ == '__main__':
    """
    argvs = sys.argv
    argc = len(argvs)
    #print argvs
    if argc < 4:
        #print("[Usage]python <image_path> <output_file_name> <limit> <gridsize>" % (argvs[0]))
        sys.exit(-1)
    """
    # Extract parameters from argvs. There are better ways to do this. Maybe using **kwargs. Change later.
    image_path = 's.jpg'
    #output_file_name = argvs[2]
    limit = 100
    grid_size = 100

    # Open the image and apply CLAHE.
    gtruth = cv2.imread(image_path)
    clahe_img = clahe_gridsize(image_path, denoise=False, verbose=False, cliplimit=int(limit), gridsize=int(grid_size))
    # Concatenate the ground truth and CLAHE processed image.
    v = np.concatenate((gtruth, clahe_img), axis=1)

    # Save the image in the given filename.
    cv2.imwrite('crop_03-11_clahe.jpg', v)
    #cv2.imshow('watershed', v)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

