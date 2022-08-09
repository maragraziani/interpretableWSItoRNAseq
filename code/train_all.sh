for GENE in SEMA5A CTTNBP2 WARS POFUT1 QPRT PLAGL2 LYX TP53RK DDX27 FSCN1 TIMP3 VAV2 RBMS1 ARSGL1 SPINK4 RAB34 MGP TNS1 SDC2 LYZ TRIB2
do
for SPLIT in split0 split1 split2 split3 split4
do
mkdir ./results/models/output_models/$GENE/
mkdir ./results/models/output_models/$GENE/$SPLIT
jbsub -mem 8g -cores 12+1 -q x86_24h -require v100 python train.py -c resnet18 -b 64 -p att -e 100 -t geneExp -g $GENE -f True -i ./data/splits/normalized/$SPLIT/ -o ./results/models/output_models/$GENE/$SPLIT/ -w  .data/TCGA-COAD/magnification_10x/
done
done
