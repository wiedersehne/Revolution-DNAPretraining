# Revolution-Pretraining
The official implementation of "Self-supervised learning for DNA sequences with Revolution networks"

## Data Availability

#### Human
To get pretraining data for Variant Effect Prediction, you need to 
1. Download hg38.fa from https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/; 
2. Use 24 chromosomes whose lengths ranges from 46 709 983 to 248 956 422. 

To get fine-tuning data for Variant Effect Prediction, you need to 
1. Download train/val/test from data/human/fine-tuning; 
2. Use vcf_Dataset in data_utils.py to create datasets for training. Sequence lengths can be adjusted. The recommended sequence length is 10kbp.

#### Plants
To get data for OCRs Prediction, you need to 
1. Download reference genome files from https://plantdeepsea-toturial2.readthedocs.io/en/latest/08-Statistics.html
   (Google Drive or Baidu Cloud Disk Drive);
2. Run ***plant_download.ipynb*** in /data/plants/plant_data/.
