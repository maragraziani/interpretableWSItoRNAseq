for gene in SDC2 TNS1 ATP9A ARMCX1 RAB34 RASD1 GNLY PLAGL2 CCDC80 B3GNT6 TP53 WARS APC AXIN2 FSCN1 CXCR4 CDHR1 ASRGL1 RBMS1 POFUT1 TIMP3 DNMT3B VAV2 DACH1 DDX27 FBN1 CAPN9 CTTNBP2
do
jbsub -mem 8g -cores 20+0 -q x86_12h python ./attention/create_attention_csvs.py $gene
done