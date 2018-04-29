get_data <- function(gdm, bw_plus_path, bw_minus_path, positions, positive, allow= NULL, n_train=25000, n_eval=1000, pdf_path= "roc_plot.pdf", plot_raw_data=TRUE, extra_enrich_bed= NULL, extra_enrich_frac= 0.1, enrich_negative_near_pos= 0.15, use_rgtsvm=FALSE, svm_type= "SVR", ncores=1, ..., debug= TRUE) {

  if(!file.exists(bw_plus_path))
    stop( paste("Can't find the bigwig of plus strand(", bw_plus_path, ")"));

  if(!file.exists(bw_minus_path))
    stop( paste("Can't find the bigwig of minus strand(", bw_minus_path, ")"));

  ########################################
  ## Divide into positives and negatives.

  batch_size = 10000;

  if (use_rgtsvm)
  {
    if(!requireNamespace("Rgtsvm"))
      stop("Rgtsvm has not been installed fotr GPU computing.");

    predict = Rgtsvm::predict.gtsvm;
    svm = Rgtsvm::svm;
  }

  #if( class(asvm)=="svm" && use_rgtsvm) class(asvm)<-"gtsvm";
  #if( class(asvm)=="gtsvm" && !use_rgtsvm) class(asvm)<-"svm";

  inter_indx <- (n_train+n_eval)
  indx_train <- c(1:n_train, (inter_indx+1):(inter_indx+n_train))
  indx_eval  <- c((n_train+1):(inter_indx), (inter_indx+n_train+1):(2*inter_indx))

  parallel_read_genomic_data <- function( x_train_bed, bw_plus_file, bw_minus_file )
  {
      interval <- unique(c( seq( 1, NROW(x_train_bed)+1, by=batch_size ), NROW(x_train_bed)+1))
      feature_list<- mclapply(1:(length(interval)-1), function(x) {
            print(paste(x, "of", length(interval)-1) );
            batch_indx<- c( interval[x]:(interval[x+1]-1) );
            return(read_genomic_data(gdm, x_train_bed[batch_indx,,drop=F], bw_plus_file, bw_minus_file));
      }, mc.cores= ncores);

	  return( do.call("rbind", feature_list) );
  }

  ## Read genomic data.
  if(debug) print("Collecting training data.")
  if(length(bw_plus_path) == 1) {
    tset <- get_test_set(positions= positions, positive= positive, allow= allow, n_samp= (n_train+n_eval), extra_enrich_bed= extra_enrich_bed, extra_enrich_frac= extra_enrich_frac, enrich_negative_near_pos= enrich_negative_near_pos)

    ## Get training indices.
    x_train_bed <- tset[indx_train,c(1:3)]
    y_train <- tset[indx_train,4]
    x_predict_bed <- tset[indx_eval,c(1:3)]
    y_predict <- tset[indx_eval,4]

	## Write out a bed of training positions to avoid during test ...
    if(debug) {
      write.table(x_train_bed, "TrainingSet.bed", quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t")
	  write.table(indx_train, "TrainIndx.Rflat")
    }

    x_train <- parallel_read_genomic_data( x_train_bed, bw_plus_path, bw_minus_path)

  } else {
    x_train <- NULL
    y_train <- NULL
    stopifnot(NROW(bw_plus_path) == NROW(bw_minus_path) & NROW(bw_plus_path) == NROW(positive))
    for(x in 1:length(bw_plus_path)){
      tset_x <- get_test_set(positions= positions[[x]], positive= positive[[x]], allow= allow[[x]], n_samp= (n_train+n_eval), extra_enrich_bed= extra_enrich_bed[[x]], extra_enrich_frac= extra_enrich_frac, enrich_negative_near_pos= enrich_negative_near_pos)

      x_train_bed <- tset_x[indx_train,c(1:3)]
      y_train <- c(y_train, tset_x[indx_train,4])

      x_train <- rbind(x_train, parallel_read_genomic_data( x_train_bed, bw_plus_path[[x]], bw_minus_path[[x]]) );
    }
  }

  gc();

  ########################################
  ## Train the model.
  if(debug) print("Fitting SVM.")
  if (svm_type == "SVR") {
    if(debug) print("Training a epsilon-regression SVR.")
    write.table(x_train,sep="\t", file="x_train")
	print("x has done")
	write.table(y_train,sep="\t",file="y_train")
	print("y has done")
  }
  if (svm_type == "P_SVM") {
    if(debug) print("Training a probabilistic SVM.")
	write.table(x_train,sep="\t",file="x_train")
	print("x has done")
	write.table(y_train,sep="\t",file="y_train")
	print("y has done")
  }

  

}


require(dREG)
library(parallel)
## Read PRO-seq data.
ps_plus_path  <- "./data/GSE66031_ac16.unt.all_plus.bw" #/usr/data/GROseq.parser/hg19/k562/proseq/
ps_minus_path <- "./data/GSE66031_ac16.unt.all_minus.bw" #/usr/data/GROseq.parser/hg19/k562/proseq/

## Get positive regions.
GROcap_tss_bed <- read.table("./data/GSE66031_ac16.dreg_peaks.bed") #"/usr/projects/GROseq.parser/tss_new/hg19.k562.new_hmm2.bed", skip=1)

## Train the SVM.
inf_positions <- get_informative_positions(ps_plus_path, ps_minus_path, depth= 0, step=50, use_ANDOR=TRUE, use_OR=FALSE) ## Get informative positions.
print(paste("Number of inf. positions: ", NROW(inf_positions)))

gdm <- genomic_data_model(window_sizes= c(10, 25, 50, 500, 5000), half_nWindows= c(10, 10, 30, 20, 20))
get_data(gdm, ps_plus_path, ps_minus_path, inf_positions, GROcap_tss_bed, pdf_path= "roc_plot.and1.lgModel.pdf", n_train=50000, n_eval=5000)
print("oye")
