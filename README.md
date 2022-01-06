# Tools and packages
- R 4.1.2 with following packages:
    - GEOquery (2.62.1)
    - tidyverse (1.3.1)
    - quantable (0.3.6)
    - ggplot2 (3.3.5)
    - sva (3.42.0)
- Python3.9.7 with following packages:
    - pandas (1.3.4)
    - numpy (1.20.3)
    - matplotlib (3.4.3)
    - scikit-learn (0.24.2)
    - joblib (1.1.0)

Note: scripts were tested using above versions of R, Python and packages. Scripts could also work with other versions, but this was not tested.

# Input data
1. Download and extract GSE98320 and GSE129166 raw series matrix files and store in Data directory
2. Download and extraxt "PrimeView.na36.annot.csv" and "HG-U133_Plus_2.na36.annot.csv" annotation files from Affymetrix website under "Current NetAffx Annotation Files" (http://www.affymetrix.com/support/technical/byproduct.affx?product=primeview & http://www.affymetrix.com/support/technical/byproduct.affx?product=hg-u133-plus) and store in Data directory
3. Execute the preprocessing.R script

# Train and test model
Execute the banff_randomforest.py script from the current directory to generate the B-HOT+ model:
python3 banff_randomforest.py --train_file Data/combat.scaled.GSE98320.csv --train_pheno Data/GSE98320.phenotype.csv --test_file Data/combat.scaled.GSE129166.biopsy.csv --test_pheno Data/GSE129166.phenotype.csv --out Data/Banff_model_performance.tsv --model Data/Banff_model.joblib --feature "bhot+"

# Pretrained model
The trained B-HOT+ model from the article can be found as "BHOT_plus_randomforest.joblib" in the Data directory.
