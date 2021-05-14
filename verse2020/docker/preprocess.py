
import argparse
from glob import glob
import os
import numpy as np
import itk
import multiprocessing


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


def smooth(image, sigma):
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    clamped = filter.GetOutput()
    return clamped


def clamp(image):
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.ClampImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetBounds(-1024, 8192)
    filter.Update()
    clamped = filter.GetOutput()
    return clamped


def process_image(filename, output_folder, sigma):
    basename = os.path.basename(filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]
    print(basename_wo_ext)
    ImageType = itk.Image[itk.SS, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(filename)
    image = reader.GetOutput()
    reoriented = reorient_to_rai(image)
    smoothed = smooth(reoriented, sigma)
    clamped = clamp(smoothed)
    clamped.SetOrigin([0, 0, 0])
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    clamped.SetDirection(m)
    clamped.Update()
    itk.imwrite(clamped, os.path.join(output_folder, basename_wo_ext + '.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser_args = parser.parse_args()
    
    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)
    
    filenames = glob(os.path.join(parser_args.image_folder, '*.nii.gz'))
    for filename in sorted(filenames):
        process_image(filename, parser_args.output_folder, parser_args.sigma)

    # pool = multiprocessing.Pool(8)
    # pool.starmap(process_image, [(filename, parser_args.output_folder, parser_args.sigma) for filename in sorted(filenames)])
