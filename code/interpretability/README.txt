## Here we analyse the model attention (of the ensembled outputs) as a means to interpret the results

To not overkill the compute needed we optimise the process by running multiple scripts and reading multiple files in parallel.

1. Create attention files by going through the output of the forward passes
Run the following to create files with summary of attention values per each split, eg. attention_values_split0.csv

> bash attention/create_attention_files.sh

2. Create easy-to-read attention records
This command creates in each gene folder a file called ensemble_attention_record.csv
The entries in each line of this file are the following:

{file_name}, {case_id}, {label}, {bin_idx}, {min_edge}, {max_edge}, {ensemble_attention_value}, {n_patches}\n


> python attention/create_attention_record.py


3. To retrieve the highest attention, I create a vector of lenght 7 (= # of gene expression bins). In this vector I store the location of the highest attention patch for that bin as I read the ensemble_attention_record.csv line by line. 
To get the top-5, I can just create 5 vectors and update them in cascade. 

The code is in > retrieve_highest_attentions.ipynb 

4. To generate the attention heatmaps maps for all input samples, run the following:
> bash generate_all_attention_maps.sh 