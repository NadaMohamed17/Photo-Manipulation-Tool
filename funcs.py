import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import matplotlib.image as img
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
import matplotlib
import matplotlib.image as img
import numpy as np
from PIL import Image, ImageTk

matplotlib.use( "TkAgg" )
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
global image1

def setappdata(image):
    image1= img.imread( image )
    return image1

def arthadd(image1, value):
    number = int( value )
    new = np.array( image1, dtype=np.float32 )
    new += number
    new = np.where( new > 255, 255, new )
    new = np.array( new, dtype="uint8" )
    image1=new
    setappdata(image1)
    return image1


def arthsubtract(image1, value):
    number = int( 30 )
    image1 = np.array( image1, dtype=np.float32 )
    image1 -= number
    image1 = np.where( image1 < 0, 0, image1 )
    image1 = np.array( image1, dtype=np.uint8 )
    setappdata(image1)
    return image1


def arthmultiply(image1, value):
    number = int( 30 )
    image1 = np.array( image1, dtype=np.float32 )
    image1 *= number
    image1 = np.where( image1 > 255, 255, image1 )
    image1 = np.array( image1, dtype=np.uint8 )
    setappdata(image1)
    return image1

def arthdivison(image1, value):
    number = int( 30 )
    image1 = np.array( image1, dtype=np.float32 )
    image1 /= number
    image1 = np.where( image1 > 255, 255, image1 )
    image1 = np.array( image1, dtype=np.uint8 )
    setappdata(image1)
    return image1


def histogram_equalization(img):
    [rows, columns] = img.shape
    num_of_pixels = rows * columns
    Histogram_img = np.zeros( img.shape, dtype='uint8' )
    freq = np.zeros( (256, 1) )
    prob = np.zeros( (256, 1) )
    output = np.zeros( (256, 1) )
    for i in range( rows ):
        for j in range( columns ):
            freq[img[i, j]] += 1
    sum = 0
    no_bins = 255
    for i in range( freq.size ):
        sum += freq[i]
        prob[i] = sum / num_of_pixels
        output[i] = np.round( prob[i] * no_bins )

        for i in range( rows ):
            for j in range( columns ):
                Histogram_img[i, j] = output[img[i, j]]
    setappdata(image1)
    return Histogram_img



def median_filter(img):
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array( img )
    for i in range( 1, rows - 1 ):
        for j in range( 1, columns - 1 ):
            temp = [
                img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                img[i, j - 1], img[i, j], img[i, j + 1],
                img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]
            ]
            temp = np.sort( temp )
            output[i, j] = temp[4]
    setappdata(image1)
    return output


def contrast_stretching(image1):
    read_image = image1
    image = np.array( read_image )
    new_image = np.array( read_image )
    s1 = 0
    s2 = 255
    r1 = np.min( read_image )
    r2 = np.max( read_image )
    rows = image.shape[0]
    columns = image.shape[1]
    range_new = (s2 - s1)
    range_old = (r2 - r1)
    for i in range( rows ):
        for j in range( columns ):
            new_image[i, j] = (range_new / range_old) * (image[i, j] - r1) + s1
    image1 = new_image.astype( np.uint8 )
    setappdata(image1)
    return image1

def Thresholding(image, k):
    image = np.array( image, dtype='uint8' )
    [rows, columns] = image.shape
    bw = np.array( image, dtype='uint8' )
    for i in range( rows ):
        for j in range( columns ):
            if image[i][j] > k:
                bw[i][j] = 0
            else:
                bw[i][j] = 1
    setappdata(image1)
    return bw


def identity(image):
    return image


def GraylevelslicingA1(image, low, high):
    image = np.array( image, dtype='uint8' )
    [rows, columns] = image.shape
    Grayslice = np.array( image, dtype='uint8' )
    for r in range( rows ):
        for c in range( columns ):
            if image[r][c] >= low and image[r][c] <= high:
                Grayslice[r][c] = 255
            else:
                Grayslice[r][c] = 0
    setappdata(image1)
    return Grayslice


def GraylevelslicingA2(image, low, high):
    image = np.array( image, dtype='uint8' )
    [rows, columns] = image.shape
    Grayslice = np.array( image, dtype='uint8' )
    for r in range( rows ):
        for c in range( columns ):
            if image[r][c] >= low and image[r][c] <= high:
                Grayslice[r][c] = 255
    setappdata(image1)
    return Grayslice



def inverseLog(img1):
    image = np.array( img1 )
    r = np.array( img1 )
    c = 255 / (np.log( 1 + np.max( img1 ) ))
    inverseLog = np.exp( img1 / c ) - 1
    inverseLog = np.array( inverseLog, dtype=np.uint8 )
    image1=inverseLog
    setappdata(image1)
    return image1

def log(imgpil):
    image = np.array( imgpil )
    r = np.array( imgpil )
    c = 255 / np.log( 1 + np.max( r ) )
    log_image = (c * np.log( 1 + r )) + 1
    log_image = np.array( log_image, dtype=np.uint8 )
    image1=log_image
    setappdata(image1)
    return image1


def Negative(image):
    neg = np.array( image, dtype='uint8' )
    L = 2 ** 8
    neg = (L - 1) - neg
    image1= neg
    setappdata(image1)
    return image1


def Power_law_transformation(img, gamma, c=1):
    img = np.array( img, dtype=np.float32 )
    y = c * (img ** gamma)
    image1=y
    setappdata(image1)
    return image1

def BHFP(img):
    image_float32 = np.float32( img )

    # do dft
    dft = cv2.dft( image_float32, flags=cv2.DFT_COMPLEX_OUTPUT )
    dft_shift = np.fft.fftshift( dft )

    # get the magnitude
    magnitude_spectrum = np.log( cv2.magnitude( dft_shift[:, :, 0], dft_shift[:, :, 1] ) )

    # initialize filter and center it
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= crow
    r -= ccol

    # make filter circular
    d = np.sqrt( np.power( r, 2.0 ) + np.power( c, 2.0 ) )
    lpFilter_matrix = np.zeros( (rows, cols, 2), np.float32 )

    # specify filter arguments
    d0 = 10
    n = 2

    # compute butterworth filter
    lpFilter = 1.0 / (1 + np.power( d0 / d, 2 * n ))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    # apply filter
    fshift = dft_shift * lpFilter_matrix

    # do idft
    f_ishift = np.fft.ifftshift( fshift )
    img_back = cv2.idft( f_ishift )

    # convert back to real output from complex
    img_back = cv2.magnitude( img_back[:, :, 0], img_back[:, :, 1] )
    image1 = img_back
    setappdata(image1)
    return image1

def BLPF(img):
    image_float32 = np.float32( img )

    # do dft
    dft = cv2.dft( image_float32, flags=cv2.DFT_COMPLEX_OUTPUT )
    dft_shift = np.fft.fftshift( dft )

    # get the magnitude
    magnitude_spectrum = np.log( cv2.magnitude( dft_shift[:, :, 0], dft_shift[:, :, 1] ) )

    # initialize filter and center it
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= crow
    r -= ccol

    # make filter circular
    d = np.sqrt( np.power( r, 2.0 ) + np.power( c, 2.0 ) )
    lpFilter_matrix = np.zeros( (rows, cols, 2), np.float32 )

    # specify filter arguments
    d0 = 10
    n = 2

    # compute butterworth filter
    lpFilter = 1.0 / (1 + np.power( d / d0, 2 * n ))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    # apply filter
    fshift = dft_shift * lpFilter_matrix

    # do idft
    f_ishift = np.fft.ifftshift( fshift )
    img_back = cv2.idft( f_ishift )

    # convert back to real output from complex
    img_back = cv2.magnitude( img_back[:, :, 0], img_back[:, :, 1] )
    image1 = img_back
    setappdata(image1)
    return image1

def GLPF(img):
    image1 = img.filter( ImageFilter.GaussianBlur )
    setappdata(image1)
    return image1
def IHPF(img):
    def highPassFiltering(img, size):  # Transfer parameters are Fourier transform spectrogram and filter size
        h, w = img.shape[0:2]  # Getting image properties
        h1, w1 = int( h / 2 ), int( w / 2 )  # Find the center point of the Fourier spectrum
        img[h1 - int( size / 2 ):h1 + int( size / 2 ), w1 - int( size / 2 ):w1 + int(
            size / 2 )] = 0  # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
        return img

    # Fourier transform
    img_dft = np.fft.fft2( img )
    dft_shift = np.fft.fftshift( img_dft )  # Move frequency domain from upper left to middle

    # High pass filter
    dft_shift = highPassFiltering( dft_shift, 200 )
    res = np.log( np.abs( dft_shift ) )

    # Inverse Fourier Transform
    idft_shift = np.fft.ifftshift( dft_shift )  # Move the frequency domain from the middle to the upper left corner
    ifimg = np.fft.ifft2( idft_shift )  # Fourier library function call
    ifimg = np.abs( ifimg )
    return  ifimg

def ILPF(img):
    M = img.shape[0]
    N = img.shape[1]
    rows = int( M / 2 )
    cols = int( N / 2 )
    mask = np.zeros( (M, N), np.uint8 )
    mask[rows - 15:rows + 15, cols - 15:cols + 15] = 1
    # conversion spatial domain to frequency
    img_fit = np.fft.fft2( img )
    img_shift = np.fft.fftshift( img_fit )
    magnitude_spectrum = 20 * np.log( np.abs( img_shift ) )
    # low pass in frequency domain
    img_shift_after_filter = img_shift * mask
    magnitude_spectrum_after_filter = magnitude_spectrum * mask
    # inverse shift
    lpf_shift = np.fft.ifftshift( magnitude_spectrum_after_filter )
    real_lpf_shift = np.fft.ifftshift( img_shift_after_filter )
    # inverse fast fourier transform
    img_back = np.fft.ifft2( real_lpf_shift )
    img_back = np.abs( img_back )
    image1= img_back
    return image1

def compositeLaplacian(img):
    gray_img = np.array( img, dtype=np.float32 )
    [rows, columns] = gray_img.shape
    mask = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    out = np.array( gray_img )
    for i in range( 1, rows - 1 ):
        for j in range( 1, columns - 1 ):
            temp = mask * gray_img[i - 1:i + 2, j - 1:j + 2]
            value = np.sum( temp )
            out[i, j] = value
    out = np.where( out < 0, 0, out )
    out = np.where( out > 255, 255, out )
    out = np.array( out, dtype='uint8' )
    image1= out
    return image1

def sobel(img):
    gray_img = np.array( img, dtype=np.float32 )
    [rows, columns] = gray_img.shape
    mask = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    out = np.array( gray_img )
    for i in range( 1, rows - 1 ):
        for j in range( 1, columns - 1 ):
            temp = mask * gray_img[i - 1:i + 2, j - 1:j + 2]
            value = np.sum( temp )
            out[i, j] = value
    out = np.where( out < 0, 0, out )
    out = np.where( out > 255, 255, out )
    out = np.array( out, dtype='uint8' )
    image1= out
    return image1

def logic_or(img):
    mask = np.ones( img.shape, dtype=np.uint8 )
    mask = cv2.circle( mask, (260, 300), 225, (255, 255, 255), -1 )
    result = cv2.bitwise_or( img, mask )
    image1 = result
    return result

def logic_and(img):
    mask = np.ones( img.shape, dtype=np.uint8 )
    mask = cv2.circle( mask, (260, 300), 225, (255, 255, 255), -1 )
    result = cv2.bitwise_and( img, mask )
    result[mask == 1]
    return result

def laplacian(img):
    gray_img = np.array( img, dtype=np.float32 )
    [rows, columns] = gray_img.shape
    mask = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    out = np.array( gray_img )
    for i in range( 1, rows - 1 ):
        for j in range( 1, columns - 1 ):
            temp = mask * gray_img[i - 1:i + 2, j - 1:j + 2]
            value = np.sum( temp )
            out[i, j] = value
    out = np.where( out < 0, 0, out )
    out = np.where( out > 255, 255, out )
    out = np.array( out, dtype='uint8' )
    image1= out
    return image1