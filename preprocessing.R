library(GEOquery)
library(tidyverse)
library(quantable)
library(ggplot2)
library(sva)

loadExpressionData <- function(filename) {
  # load datasets from series matrix formats
  # find start and end locations of the actual gene expression matrix
  start_pattern <- "!series_matrix_table_begin"
  end_pattern <- "!series_matrix_table_end"
  GEO_lines <- readLines(filename)
  GEO_start <- grep(start_pattern, GEO_lines)
  GEO_nrows <- grep(end_pattern, GEO_lines) - GEO_start - 2
  # read gene expression matrix
  GEO <- read.table(filename, header=T, sep="", skip=GEO_start, nrows=GEO_nrows)
  GEO$ID_REF <-gsub('"','',GEO$ID_REF)
  return(GEO)
}

loadMetaData <- function(filename) {
  # load datasets from series matrix formats
  # find start and end locations of the actual gene expression matrix
  start_pattern <- "!Sample_title"
  end_pattern <- "!series_matrix_table_begin"
  GEO_lines <- readLines(filename)
  meta_start <- grep(start_pattern, GEO_lines)
  meta_nrows <- grep(end_pattern, GEO_lines) - meta_start - 2
  # read gene expression matrix
  meta_df <- read.table(filename, header=T, sep="", skip=meta_start, nrows=meta_nrows)
  rownames(meta_df) = make.names(meta_df$X.Sample_geo_accession, unique=TRUE)
  return(as.data.frame(t(meta_df)))
}

matchAggregate <- function(expression, mapping, id_type) {
  expression$Probe.Set.ID <- expression$ID_REF
  expression_mapped <- left_join(expression, mapping, by = "Probe.Set.ID")
  # aggregate expression by their median values
  expression_aggre = aggregate(subset(expression_mapped, select=-c(ID_REF,Probe.Set.ID)),by = list(expression_mapped[[id_type]]), FUN =function(x) x=median(x) )
  names(expression_aggre)[names(expression_aggre) == 'Group.1'] <- id_type
  return(expression_aggre)
}

getProbeMapping <- function(probe_annotation_file, identifier_type) {
  probe_annot <- read.table(probe_annotation_file, header=T, sep=",", skip=25)
  probe_mapping <- probe_annot[, c("Probe.Set.ID", identifier_type)]
  probe_mapping <- separate_rows(probe_mapping, identifier_type, sep=" /// ")
  return(probe_mapping)
}

combineDataSets <- function(dataset1, dataset2){
  shared.genes <- intersect(dataset1[["Entrez.Gene"]], dataset2[["Entrez.Gene"]])
  dataset1 <- dataset1[dataset1[["Entrez.Gene"]] %in% shared.genes,]
  dataset2 <- dataset2[dataset2[["Entrez.Gene"]] %in% shared.genes,]
  rownames(dataset1) <- dataset1[["Entrez.Gene"]]
  rownames(dataset2) <- dataset2[["Entrez.Gene"]]
  combined.dataset <- cbind(dataset1, dataset2)
  combined.dataset <- combined.dataset[ , -which(names(combined.dataset) %in% c("X", "Entrez.Gene", "ID_REF", "Probe.Set.ID"))]
  return(combined.dataset)
}

normalizeScaleDatasets <- function(combined.dataset, batch.indication){
  combat.combined <- ComBat(dat=combined.dataset, batch=batch.indication, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
  dataset.names <- unique(batch.indication)
  n.datasets <- length(dataset.names)
  datasets <- list()
  for(i in 1:n.datasets) {
    dataset <- combat.combined[, which(batch.indication == dataset.names[[i]])]
    dataset.scaled <- robustscale(dataset, dim = 1)$data
    datasets[[i]] <- dataset.scaled
  }
  return(datasets)
}

# set script directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# get probe mapping
primeview_entrez_mapping <- getProbeMapping("Data/PrimeView.na36.annot.csv", "Entrez.Gene")
hgu133plus2_entrez_mapping <- getProbeMapping("Data/HG-U133_Plus_2.na36.annot.csv", "Entrez.Gene")

# load expression data 
GSE98320 <- loadExpressionData("Data/GSE98320_series_matrix.txt")
GSE129166 <- loadExpressionData("Data/GSE129166_series_matrix.txt")

# load metadata
GSE98320_meta <- loadMetaData("Data/GSE98320_series_matrix.txt")
GSE129166_meta <- loadMetaData("Data/GSE129166_series_matrix.txt")

# determine NR, ABMR and TCMR labels from GSE98320
GSE98320_meta$Label <- with(GSE98320_meta, ifelse(
  X.Sample_characteristics_ch1.1 == "cluster (1,2,3,4,5,6): 1", "NR", ifelse(
    X.Sample_characteristics_ch1.1 == "cluster (1,2,3,4,5,6): 4" | X.Sample_characteristics_ch1.1 == "cluster (1,2,3,4,5,6): 5" | X.Sample_characteristics_ch1.1 == "cluster (1,2,3,4,5,6): 6", "ABMR", ifelse(
      X.Sample_characteristics_ch1.1 == "cluster (1,2,3,4,5,6): 2", "TCMR", ""
    ))))
GSE98320_meta <- GSE98320_meta[GSE98320_meta$Label %in% c("NR", "ABMR", "TCMR"),]

# retrieve biopsy samples only from GSE129166
GSE129166_meta <- GSE129166_meta[GSE129166_meta$X.Sample_characteristics_ch1 == "tissue: kidney allograft biopsy",]

# determine NR, ABMR and TCMR samples from GSE129166
GSE129166_meta$Label <- with(GSE129166_meta, ifelse(
  X.Sample_characteristics_ch1.1 == "tcmr (no: 0_borderline:1_TCMR:2): 0" & X.Sample_characteristics_ch1.2 == "abmr (no: 0_Yes:1): 0", 'NR', ifelse(
    X.Sample_characteristics_ch1.1 == "tcmr (no: 0_borderline:1_TCMR:2): 0" & X.Sample_characteristics_ch1.2 == "abmr (no: 0_Yes:1): 1", 'ABMR', ifelse(
      X.Sample_characteristics_ch1.1 == "tcmr (no: 0_borderline:1_TCMR:2): 2" & X.Sample_characteristics_ch1.2 == "abmr (no: 0_Yes:1): 0", 'TCMR', ""
    ))))
GSE129166_meta <- GSE129166_meta[GSE129166_meta$Label %in% c("NR", "ABMR", "TCMR"),]

# retrieve necessary phenotype information from metadata table and write to file
GSE98320_pheno <- data.frame("Sample_id" = rownames(GSE98320_meta), "Label" = GSE98320_meta$Label)
GSE129166_pheno <- data.frame("Sample_id" = rownames(GSE129166_meta), "Label" = GSE129166_meta$Label)
write.table(GSE98320_pheno, file="Data/GSE98320.phenotype.csv", row.names=F, col.names=T, quote=F, sep=",")
write.table(GSE129166_pheno, file="Data/GSE129166.phenotype.csv", row.names=F, col.names=T, quote=F, sep=",")

# separate biopsy from blood data and filter expression on selected phenotypes
GSE129166_biopsies <- GSE129166[, colnames(GSE129166) %in% c(GSE129166_pheno$Sample_id, "ID_REF")]
GSE98320 <- GSE98320[, colnames(GSE98320) %in% c(GSE98320_pheno$Sample_id, "ID_REF")]

# match expression data with BHOT genes and aggregate them by entrez id
GSE98320_aggre <- matchAggregate(GSE98320, primeview_entrez_mapping, "Entrez.Gene")
GSE129166_biopsies_aggre <- matchAggregate(GSE129166_biopsies, hgu133plus2_entrez_mapping, "Entrez.Gene")

# combine the different datasets for removal of batch effects by array type
GSE98320_GSE129166_biopsies <- combineDataSets(GSE98320_aggre, GSE129166_biopsies_aggre)

# execute combat & separate scaling
combat_batch <- rep(c("GSE98320", "GSE129166.biopsies"), times=c(nrow(GSE98320_pheno), nrow(GSE129166_pheno)))
datasets <- normalizeScaleDatasets(as.matrix(GSE98320_GSE129166_biopsies), combat_batch)
combined_scaled_datasets <- as.data.frame(t(do.call(cbind, datasets)))

# separate datasets again and save data
combat_scaled_GSE98320 <- combined_scaled_datasets[GSE98320_pheno$Sample_id, ]
combat_scaled_GSE129166_biopsy <- combined_scaled_datasets[GSE129166_pheno$Sample_id, ]
write.table(combat_scaled_GSE98320, file="Data/combat.scaled.GSE98320.csv", row.names=T, col.names=T, quote=F, sep=",")
write.table(combat_scaled_GSE129166_biopsy, file="Data/combat.scaled.GSE129166.biopsy.csv", row.names=T, quote=F, col.names=T, sep=",")