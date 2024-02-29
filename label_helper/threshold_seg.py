import os
import sys
sys.path.append("..")
import SimpleITK  as sitk
import numpy as np
from utils.file_manage import read_all_file_paths_from_dir


def ostu_threshold_seg(img_path):
    img = sitk.ReadImage(img_path)
    background = sitk.OtsuThreshold(img)
    seg_array = 1-sitk.GetArrayFromImage(background)

    seg = sitk.GetImageFromArray(seg_array)
    seg.CopyInformation(img)
    return seg


def remove_noise(seg):

    # get img centroid and its label
    img_size = seg.GetSize()
    center = (img_size[0] // 2, img_size[1] // 2, img_size[2] // 2)
    

    # only remain the connected component with centroid
    cc_filter = sitk.ConnectedComponentImageFilter()
    # cc_filter.SetFullyConnected(True) #3D
    cc_img = cc_filter.Execute(seg)
    center_label = cc_img[center]

    cleaned_img = sitk.BinaryThreshold(cc_img, center_label, center_label, 1, 0)
    return cleaned_img



def main():
    img_paths = read_all_file_paths_from_dir('/home/hanruishi/tantan_preprocess/crop/cta_rm_bone/')
    output_folder = '/home/hanruishi/tantan_preprocess/crop/pre_seg/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for img_path in img_paths:
        yen_seg = ostu_threshold_seg(img_path)
        clean_seg = remove_noise(yen_seg)

        save_path = output_folder +'seg_' + os.path.basename(img_path) 
        sitk.WriteImage(clean_seg,save_path)


if __name__ == "__main__":
    main()





