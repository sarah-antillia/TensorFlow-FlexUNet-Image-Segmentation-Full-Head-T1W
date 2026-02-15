# Copyright 2026 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2026/02/13 ImageMaskDatasetGenerator.py

import os
import sys
import io
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import cv2

class ImageMaskDatasetGenerator:

  def __init__(self, 
               resize     = 512,
               augmentation=True):
  
    self.index   = 1000
    self.RESIZE  = (resize, resize)
    self.seed = 137
     
    self.file_format= ".png"


  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    image_files = sorted(glob.glob(images_dir + "/*.nii"))
    mask_files  = sorted(glob.glob(masks_dir  + "/*.nii"))
    self.output_images_dir = output_images_dir
    self.output_masks_dir  = output_masks_dir
    

    l1 = len(mask_files)
    l2 = len(image_files)
    print("--- l1: {} l2: {}".format(l1, l2))
    
    if l1 != l2:
      raise Exception("Unmatched number of seg_files and image_files ")
    for i in range(l1):
      self.index += 1

      self.generate_mask_files(mask_files[i],  self.index, output_masks_dir) 
      self.generate_image_files(image_files[i],self.index, output_images_dir)

  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    if scale == 0:
      scale +=  1
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  def generate_image_files(self, niigz_file, index, output_images_dir):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
   
    print("=== image shape {}".format(fdata.shape))
    (w, h, d,) = fdata.shape
    for i in range(d):
      img = fdata[:,:,i,]
      print("Image shape", img.shape)
      
      filename  = str(index) + "_" + str(i) + self.file_format
      filepath  = os.path.join(output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
    
        img  *= 255
        img  = self.normalize(img)   
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(filepath, img)
        print("=== Saved {}".format(filepath))
 
      else:
        print("=== Skipped {}".format(filepath))

  def generate_mask_files(self, niigz_file, index, output_masks_dir):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    print("=== mask shape {}".format(fdata.shape))
    
    (w, h, d) = fdata.shape
    for i in range(d):
      img = fdata[:,:,i]
      filename  = str(index) +"_" + str(i) + self.file_format
      filepath  = os.path.join(output_masks_dir, filename)
      
      if img.any() >0:
        
        img = self.colorize_mask(img)
        img = img.astype('uint8')
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filepath, img)
      else:
        print("=== Skipped {}".format(filepath))

  def colorize_mask(self, mask):
    # initialize output to zeros
    h, w, = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.float32)
    """
    1: Air cavities:green
    2: White matter (WM) :white
    3: Gray matter (GM)
    4: Cerebrospinal fluid (CSF): cyan
    5: Bone : yellow
    6: Non-brain soft tissue skin: (128,20,20)
    """
    #              1,         2,           3,            4,           5,         6
    rgb_colors = [(10,180,10),(255,255,255),(128,128,128), (0,128,255),(255,255,0),(180,100,100),] #(128,128,128),(255,255,255)]
    index = 1
    for rgb_color in rgb_colors:
      (r, g, b) = rgb_color
      colorized[np.equal(mask, index)] = (b, g, r)
      index += 1
    
    return colorized


"""
Full-Head MRI & Segmentation of Stroke Patients
https://www.kaggle.com/datasets/andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients/data

https://arxiv.org/pdf/2501.18716
Full-Head Segmentation of MRI with Abnormal Brain
Anatomy: Model and Data Release

"""
if __name__ == "__main__":
  try:
    images_dir = "./Data/Anonymized_Subjects/T1-Weighted MRI/"

    masks_dir  = "./Data/Anonymized_Subjects/Full-Head Segmentation/"

    output_dir = "./Full-Head-T1W-master/"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")

    os.makedirs(output_images_dir)  
    os.makedirs(output_masks_dir)

    generator = ImageMaskDatasetGenerator()
    
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

 
  except:
    traceback.print_exc()

 
