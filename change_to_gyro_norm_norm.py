import glob
import json
import re

files = glob.glob("./data/*")
for file in files:
    with open(file) as f:
        read_dict = json.load(f)
    chest_gyro_list,pocket_gyro_list = read_dict['chest_gyro'],read_dict['pocket_gyro']
    output_chest_gyro_list,output_pocket_gyro_list = [],[]
    for line in chest_gyro_list:
        gyro_norm = sum([i**2 for i in line])/len(line)
        output_chest_gyro_list+= [gyro_norm]
    for line in pocket_gyro_list:
        gyro_norm = sum([i**2 for i in line])/len(line)
        output_pocket_gyro_list += [gyro_norm]
    read_dict['chest_gyro'] = output_chest_gyro_list
    read_dict['pocket_gyro'] = output_pocket_gyro_list

    save_file = file.replace("data","fft")
    with open(save_file,"w") as f:
        json.dump(read_dict,f)
    
