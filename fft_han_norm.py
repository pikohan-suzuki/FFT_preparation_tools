import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import re
import json

def fft_han_norm(acc_list,s_rate=60):
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
    w1 = signal.hann(N)
    f = f * w1
    F = np.fft.fft(f,n=N)
    F = F / (N / 2)
    F[0] = F[0] /2
    F = F[:N//2]
    F = np.abs(F)
    F[:2] = 0.

    freq = np.arange(0,s_rate//2,s_rate/N)

    # normalization
    max_F = max(F)
    F_norm = [i/max_F for i in F]

    ### low cut
    cut_per = 0.1
    cut_value = max(F_norm) * cut_per
    F_norm = [0 if i < cut_value else i for i in F_norm]
    return freq,F_norm

if __name__ == "__main__":
    ## The location of directry that contains the input files.
    files = glob.glob("./data/untouched/*")
    for i,file in enumerate(files):
        output_dict = dict()
        with open(file) as f:
            lines = f.readlines()
        acc_lines = [float(line.split(",")[1]) for line in lines]
        freq,fft_list = fft_han_am(acc_lines)
        # freq,fft_list = fft_han_norm(acc_lines)
        output_dict["acc"] = fft_list

        if len(lines[0].split(",")) > 2:
            gyro_list = [[ float(gyro_i) for gyro_i in line.split(",")[2:5] ] for line in lines]
            output_dict["gyro"] = gyro_list

        file_name = re.split("[\\\|/]", file)[-1].split("_")
        file_name = file_name[0] + "_" + file_name[1] + "_" + file_name[-1]
        # if i % 20 == 0:
        #     fig = plt.figure()
        #     plt.plot(freq,fft_list)
        #     plt.title(file)
        #     fig.savefig("".format(file_name))
        #     plt.close()

        # write_text =""
        # for line in fft_list:
        #     write_text += str(line)
        #     write_text += "\n"
        
        save_dir = "data/fft"
        with open("{}/{}".format(save_dir,file_name),"w") as f:
            # f.write(write_text[:-1])
            json.dump(output_dict,f)