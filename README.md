# DIMO-VAE
## Environment
DILCR uses the Paddle deep learning framework.
If you want to run DILCR, please ensure that the Paddle framework is already installed on this machine. All the specific packages required can be found in [requirements.txt](requirements.txt)

## Dataset
The 10 cancer multi-omics data and patient clinical data used in this article can be obtained through the TCGA public platform \url{https://portal.gdc.cancer.gov}. All the multi-omics data, survival data, and clinical data of patients in this experiment are taken from \url{http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html} \cite{rappoport2018multi}. The BRCA PAM50 molecular typing used in this paper is obtained by R the package TCGAbiolinks \cite{colaprico2016tcgabiolinks} was obtained.

## Run
Firstly, you need to add the DATASET_PATH in main.py. Set to the file path where your data is located. Then directly run the main.py file after pouring in the relevant package.

## Future
Soon we will provide a Pytorch version of DILCR so that everyone can reproduce it better.
