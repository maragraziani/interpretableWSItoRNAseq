#%matplotlib inline
#import matplotlib.pyplot as plt
import os
import numpy as np
def assign_case_to_bin(case_label, bin_edges):
    i=0
    while i<len(bin_edges)-1:
        if case_label <= bin_edges[i+1] and case_label >= bin_edges[i]:
            return i, bin_edges[i], bin_edges[i+1]
        else:
            i+=1

gene_list= [
'MGP']
for gene in gene_list:
    csvs={}
    splits= ['split1', 'split2', 'split3']#,'split4']
    for split in splits:
        #print(split)
        FOLDER='./data/intermediate_results/models/output_models/{}/{}/'.format(gene,split)
        file = open("{}attention_values_{}.csv".format(FOLDER,split),'r')
        csvs[split]=file
    test_images_list = open("{}attention_cases_{}.csv".format(FOLDER,split),'r').readlines()
    ensemble_attention_record = open("./results/intermediate_results/models/output_models/{}/ensemble_attention_record.csv".format(gene), 'w')
    labels_file_lines = open("./data/splits/normalized/test/{}_test.csv".format(gene),"r").readlines()
    n_patches_per_case=np.load("./data/test_files_infolders.npy", allow_pickle=True).item()

    cases=[]
    labels=[]
    labels_dic={}
    for line in labels_file_lines:
        case=line.split(',')[0]
        cases.append(case)
        labels_dic[case]=float(line.split(', ')[1].strip('\n'))
        labels.append(float(line.split(', ')[1].strip('\n')))
        hist, bin_edges = np.histogram(labels, bins=7)

    values=np.zeros(86830)

    for file_index in range(0,86830):
        for split in splits:
            values[file_index]+=float(csvs[split].readline())
        values[file_index]/=len(splits)

    for file_index in range(0,86830):
        file_name = test_images_list[file_index].strip('\n')
        case_id = file_name.split('10x/')[1]
        case_id = case_id[:12]
        label = labels_dic[case_id]
        bin_idx, min_edge, max_edge = assign_case_to_bin(label, bin_edges)
        ensemble_attention_value = values[file_index]
        n_patches = n_patches_per_case[case_id]
        ensemble_attention_record.write(f"{file_name}, {case_id}, {label}, {bin_idx}, {min_edge}, {max_edge}, {ensemble_attention_value}, {n_patches}\n")

    ensemble_attention_record.close()
    for k in csvs.keys():
        csvs[k].close()
