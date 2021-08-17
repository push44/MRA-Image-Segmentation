import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from openpyxl import load_workbook
import openpyxl
from skimage.transform import rescale, resize, downscale_local_mean

counter_writing_one = 0
counter_writing_zer_or = 0
counter_writing_zer_th_or = 0

number_of_cases = 39
path_images = './Original_Images_Binary/'
path_segmentations = './Segmentations_Binary/'
path_output = './Patches/'

output_3d_volume_size_high = [32, 32, 32]
output_3d_volume_size_low = [64, 64, 64]
output_3d_volume_size_low_true = [32, 32, 32]
output_size = [32, 32, 32]
print(output_3d_volume_size_low_true)

count1 = 0
count0 = 0

wb = load_workbook(filename = './Original_Images/ICAD_MRA_Segmentation_Worksheet.xlsx')
sheet_ranges = wb['Corrected']

threshold = 115
th = 0.8

for m in np.arange(2,number_of_cases+2):
	a = 'A' + str(m)
	c = sheet_ranges[a].value

	for l in os.listdir(path_images):
		if (c + '_Binary') == l:
			Case = c
			original_bounded = np.load(path_images + Case + '_Binary')
			voxelized_bounded = np.load(path_segmentations + Case + '_Segmentation_Binary')
			
			cutting_len = np.int((output_3d_volume_size_low[0]-output_size[0])/2)
			cutting_len_high = np.int((output_3d_volume_size_high[0]-output_size[0])/2)

			print(np.shape(original_bounded))
			print(np.shape(voxelized_bounded))

			zer = []
			zer_or = []
			zer_th_or = []
			one = []
			for i in range((original_bounded.shape[0]-2*cutting_len)//output_size[0]):
				for j in range((original_bounded.shape[1]-2*cutting_len)//output_size[1]):
					for k in range((original_bounded.shape[2]-2*cutting_len)//output_size[2]):
						original_bounded_cut = original_bounded[(i*output_size[0]+cutting_len):((i+1)*output_size[0]+cutting_len),(j*output_size[1]+cutting_len):((j+1)*output_size[1]+cutting_len),(k*output_size[2]+cutting_len):((k+1)*output_size[2]+cutting_len)]
						voxelized_bounded_cut = 1*(voxelized_bounded[(i*output_size[0]+cutting_len):((i+1)*output_size[0]+cutting_len),(j*output_size[1]+cutting_len):((j+1)*output_size[1]+cutting_len),(k*output_size[2]+cutting_len):((k+1)*output_size[2]+cutting_len)] > th)
						if ((np.asarray(voxelized_bounded_cut)>0).sum()>0):
							one.append([i,j,k])
						else:
							zer.append([i,j,k])
							if ((np.asarray(original_bounded_cut*255)>threshold).sum()>0):
								zer_th_or.append([i,j,k])
							else:
								zer_or.append([i,j,k])
			print(Case)
			print(np.asarray(one).shape)
			print(np.asarray(zer_th_or).shape)
			print(np.asarray(zer_or).shape)

			for i in range(np.asarray(one).shape[0]):
				v = np.asarray(1*(voxelized_bounded>th))
				#print(v)
				o = np.asarray(original_bounded)
				one_arr = np.asarray(one)
				#Write true output labels
				out_file = open(path_output + 'Voxel_' + str(output_size[0]) + '/ones_out/' + str(counter_writing_one),'wb')
				np.save(out_file,v[np.int(one_arr[i,0]*output_size[0]+cutting_len):np.int((one_arr[i,0]+1)*output_size[0]+cutting_len),np.int(one_arr[i,1]*output_size[1]+cutting_len):np.int((one_arr[i,1]+1)*output_size[1]+cutting_len),np.int(one_arr[i,2]*output_size[2]+cutting_len):np.int((one_arr[i,2]+1)*output_size[2]+cutting_len)])
				#Write High resolution input
				out_file = open(path_output + 'Orig_high_' + str(output_3d_volume_size_high[0]) + '/ones_high/' + str(counter_writing_one),'wb')
				x_low=np.int(one_arr[i,0]*output_size[0]+0.5*output_3d_volume_size_low[0]-0.5*output_3d_volume_size_high[0])
				x_high=np.int(one_arr[i,0]*output_size[0]+0.5*output_3d_volume_size_low[0]+0.5*output_3d_volume_size_high[0])
				y_low=np.int(one_arr[i,1]*output_size[1]+0.5*output_3d_volume_size_low[1]-0.5*output_3d_volume_size_high[1])
				y_high=np.int(one_arr[i,1]*output_size[1]+0.5*output_3d_volume_size_low[1]+0.5*output_3d_volume_size_high[1])
				z_low=np.int(one_arr[i,2]*output_size[2]+0.5*output_3d_volume_size_low[2]-0.5*output_3d_volume_size_high[2])
				z_high=np.int(one_arr[i,2]*output_size[2]+0.5*output_3d_volume_size_low[2]+0.5*output_3d_volume_size_high[2])
				np.save(out_file,o[x_low:x_high,y_low:y_high,z_low:z_high])
				#Write Low resolution input
				out_file = open(path_output + 'Orig_low_' + str(output_3d_volume_size_low[0]) + '/ones_low/' + str(counter_writing_one),'wb')
				x_low=np.int(one_arr[i,0]*output_size[0]+0.5*output_3d_volume_size_low[0]-0.5*output_3d_volume_size_low[0])
				x_high=np.int(one_arr[i,0]*output_size[0]+0.5*output_3d_volume_size_low[0]+0.5*output_3d_volume_size_low[0])
				y_low=np.int(one_arr[i,1]*output_size[1]+0.5*output_3d_volume_size_low[1]-0.5*output_3d_volume_size_low[1])
				y_high=np.int(one_arr[i,1]*output_size[1]+0.5*output_3d_volume_size_low[1]+0.5*output_3d_volume_size_low[1])
				z_low=np.int(one_arr[i,2]*output_size[2]+0.5*output_3d_volume_size_low[2]-0.5*output_3d_volume_size_low[2])
				z_high=np.int(one_arr[i,2]*output_size[2]+0.5*output_3d_volume_size_low[2]+0.5*output_3d_volume_size_low[2])
				np.save(out_file,resize(o[x_low:x_high,y_low:y_high,z_low:z_high],(output_3d_volume_size_low_true[0],output_3d_volume_size_low_true[1],output_3d_volume_size_low_true[2])))
				
				counter_writing_one = counter_writing_one + 1
			'''
			for i in range(np.asarray(zer_th_or).shape[0]):
				v = np.asarray(1*(voxelized_bounded>th))
				o = np.asarray(original_bounded)
				one_arr = np.asarray(zer_th_or)
				out_file = open(path + 'Voxel_' + str(output_3d_volume_size[0]) + '/zeros_th_115/' + str(counter_writing_zer_th_or),'wb')
				np.save(out_file,v[np.int(one_arr[i,0]*output_3d_volume_size[0]):np.int((one_arr[i,0]+1)*output_3d_volume_size[0]),np.int(one_arr[i,1]*output_3d_volume_size[1]):np.int((one_arr[i,1]+1)*output_3d_volume_size[1]),np.int(one_arr[i,2]*output_3d_volume_size[2]):np.int((one_arr[i,2]+1)*output_3d_volume_size[2])])
				out_file = open(path + 'Orig_' + str(output_3d_volume_size[0]) + '/zeros_th_115/' + str(counter_writing_zer_th_or),'wb')
				np.save(out_file,o[np.int(one_arr[i,0]*output_3d_volume_size[0]):np.int((one_arr[i,0]+1)*output_3d_volume_size[0]),np.int(one_arr[i,1]*output_3d_volume_size[1]):np.int((one_arr[i,1]+1)*output_3d_volume_size[1]),np.int(one_arr[i,2]*output_3d_volume_size[2]):np.int((one_arr[i,2]+1)*output_3d_volume_size[2])])
				counter_writing_zer_th_or = counter_writing_zer_th_or + 1

			for i in range(np.asarray(zer_or).shape[0]):
				v = np.asarray(1*(voxelized_bounded>th))
				o = np.asarray(original_bounded)
				one_arr = np.asarray(zer_or)
				out_file = open(path + 'Voxel_' + str(output_3d_volume_size[0]) + '/zeros/' + str(counter_writing_zer_or),'wb')
				np.save(out_file,v[np.int(one_arr[i,0]*output_3d_volume_size[0]):np.int((one_arr[i,0]+1)*output_3d_volume_size[0]),np.int(one_arr[i,1]*output_3d_volume_size[1]):np.int((one_arr[i,1]+1)*output_3d_volume_size[1]),np.int(one_arr[i,2]*output_3d_volume_size[2]):np.int((one_arr[i,2]+1)*output_3d_volume_size[2])])
				out_file = open(path + 'Orig_' + str(output_3d_volume_size[0]) + '/zeros/' + str(counter_writing_zer_or),'wb')
				np.save(out_file,o[np.int(one_arr[i,0]*output_3d_volume_size[0]):np.int((one_arr[i,0]+1)*output_3d_volume_size[0]),np.int(one_arr[i,1]*output_3d_volume_size[1]):np.int((one_arr[i,1]+1)*output_3d_volume_size[1]),np.int(one_arr[i,2]*output_3d_volume_size[2]):np.int((one_arr[i,2]+1)*output_3d_volume_size[2])])
				counter_writing_zer_or = counter_writing_zer_or + 1
			'''

#count1 = (voxelized[(np.abs(np.int(label[3]))+np.int(0.5*(output_3d_volume_size[0]+1))):(np.abs(np.int(label[0]))-np.int(0.5*(output_3d_volume_size[0]-1))),(np.abs(np.int(label[4]))+np.int(0.5*(output_3d_volume_size[1]+1))):(np.abs(np.int(label[1]))-np.int(0.5*(output_3d_volume_size[1]-1))),(np.abs(np.int(label[2]))+np.int(0.5*(output_3d_volume_size[2]+1))):(np.abs(np.int(label[5]))-np.int(0.5*(output_3d_volume_size[2]-1)))] == 255).sum()
#count0 = (voxelized[(np.abs(np.int(label[3]))+np.int(0.5*(output_3d_volume_size[0]+1))):(np.abs(np.int(label[0]))-np.int(0.5*(output_3d_volume_size[0]-1))),(np.abs(np.int(label[4]))+np.int(0.5*(output_3d_volume_size[1]+1))):(np.abs(np.int(label[1]))-np.int(0.5*(output_3d_volume_size[1]-1))),(np.abs(np.int(label[2]))+np.int(0.5*(output_3d_volume_size[2]+1))):(np.abs(np.int(label[5]))-np.int(0.5*(output_3d_volume_size[2]-1)))] < 255).sum()

#print('Count for ones: ' + str(count1))

#print('Count for zeros: ' + str(count0))
