import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import re
import json

def fft_han_am(acc_list,s_rate=60):    
    i=0
    while len(acc_list) > 2**i:
        i+=1
    N = 2**i
    avg = sum(acc_list)/len(acc_list)
    blank = N - len(acc_list)
    bef = blank//2
    af = blank -bef
    f = [avg]*bef + acc_list + [avg]*af
    f = [i-avg for i in f]

    cut_len = N//2
    num_of_cut = 3
    cut_list = [[]] * num_of_cut
    for i in range(num_of_cut):
        start_i = cut_len//2*i
        f_cut = f[start_i:start_i+cut_len]
        w_han = signal.hann(cut_len)
        f_han = f_cut * w_han
        F = np.fft.fft(f_han,n=cut_len)
        F = F / (cut_len / 2)
        F[0] = F[0] /2
        F = F[:cut_len//2]
        F = np.abs(F)
        F[:2] = 0.
        cut_list[i] = F

    freq = np.arange(0,s_rate//2,s_rate/cut_len)
    fft_list = [0.] * (cut_len//2)

    # add each fft values
    for i in range(num_of_cut):
        for j in range(cut_len//2):
            fft_list[j] += cut_list[i][j]
    # get a-means 
    for i in range(len(fft_list)):
        fft_list[i] = fft_list[i]/num_of_cut


    # normalization
    max_F = max(fft_list)
    fft_list = [i/max_F for i in fft_list]

    ### low cut
    cut_per = 0.1
    cut_value = max(fft_list) * cut_per
    fft_list = [0 if i < cut_value else i for i in fft_list]

    return freq,fft_list

if __name__ == "__main__":
    ## The location of directry that contains the input files.
    files = glob.glob("./data/untouched/*")
    for i,file in enumerate(files):
        output_dict = dict()
        with open(file) as f:
            lines = f.readlines()
        acc_lines = [float(line.split(",")[1]) for line in lines]
        freq,fft_list = fft_han_am(acc_lines)
        output_dict["acc"] = fft_list

        if len(lines[0].split(",")) > 2:
            gyro_list = [[ float(gyro_i) for gyro_i in line.split(",")[2:5] ] for line in lines]
            output_dict["gyro"] = gyro_list

        file_name = re.split("[\\\|/]", file)[-1]
        label = "_".join(file_name.split("_")[1:-1])
        output_dict["label"] = label
        output_dict["type"] = file_name.split("_")[0]
        
        save_dir = "data/fft"
        with open("{}/{}".format(save_dir,file_name),"w") as f:
            json.dump(output_dict,f)