library(devtools)
library(reticulate)
use_condaenv("sceasy", required = TRUE)
library(Seurat)
library(SeuratDisk)
library(microbenchmark)
library(zellkonverter)
library(sceasy)
library(pryr)
library(ALRA)
library(rhdf5)


benchmark_alra_r <- function(input_path, cell_counts,num_runs,output_dir) {
  # Convert the .h5ad file to .h5seurat format
  # ad <- readH5AD(h5ad_file)
  # print(ad)
  # seurat_obj <- as.Seurat(ad, counts = "X", data = NULL)
  # h5seurat_file <- sub("\\.h5ad$", ".h5seurat", h5ad_file)
  # Convert(h5ad_file, dest = "h5seurat", overwrite = TRUE, misc = FALSE)

  # # Load the .h5seurat file into a Seurat object
  # seurat_obj <- LoadH5Seurat(h5seurat_file)

  # print(seurat_obj)

  cpu_hours <- c()
  memory_usage <- c()
  execution_time <- c()
  k_values <- c()
  percent_non_zero <- c()

# Effectuer l'analyse pour chaque nombre de cellules
  for (n_cells in cell_counts) {

    h5ad_file <- file.path(input_path, paste0("E-MTAB-8142_", n_cells, "_cells.h5ad"))
    ad <- readH5AD(h5ad_file)
    seurat_obj <- as.Seurat(ad, counts = "X", data = NULL)
    rm(ad)
    gc()

    for (run in 1:num_runs) {
      random_state <- rnorm(1)
      # Normalized matrix as a transposed matrix from Seurat
      A_norm <- t(as.matrix(seurat_obj@assays$originalexp@data))
        
        
      # Start memory monitoring
      start_memory <- mem_used() / (1024^2)
      start_cpu_time <- proc.time()["user.self"] + proc.time()["sys.self"]
        
      # Run ALRA
      benchmark <- microbenchmark(
        {
          k_choice <- choose_k(A_norm,mkl.seed=random_state)
          result.completed <- alra(A_norm, k = k_choice$k,,mkl.seed=random_state)
          A_norm_completed <- result.completed[[3]]  # Matrice complétée
        }, times = 1
      )
        
      # Stop monitoring
      end_memory <- mem_used() / (1024^2)
      end_cpu_time <- proc.time()["user.self"] + proc.time()["sys.self"]
        
      # Calculer les pourcentages de gènes non-zéro
      num_non_zero <- sum(A_norm_completed != 0)
      total_genes <- length(A_norm_completed)
      non_zero_percentage <- (num_non_zero / total_genes) * 100

      # Enregistrer les résultats
      cpu_time_seconds <- end_cpu_time - start_cpu_time
      cpu_time_hours <- cpu_time_seconds / 3600
      cpu_hours <- c(cpu_hours, cpu_time_hours)
      memory_usage <- c(memory_usage, end_memory - start_memory)
      execution_time <- c(execution_time, median(benchmark$time / 1e9))  # Temps en secondes
      k_values <- c(k_values, k_choice$k)
      percent_non_zero <- c(percent_non_zero, non_zero_percentage)

      csv_file <- file.path(output_dir, paste0("ALRA_completed_run_",run, "_cells",n_cells,".csv"))
      write.csv(A_norm_completed, csv_file, row.names = FALSE)

      rm(A_norm, A_norm_completed, result.completed#,seurat_obj
      )
      gc()
    }
  }
  
  # Créer un tableau avec les résultats
  results_df <- data.frame(
  Run = rep(1:num_runs, times = length(cell_counts)),
  Number_of_Cells = rep(cell_counts, each = num_runs),
  k = k_values,
  Percent_Non_Zero = percent_non_zero,
  CPU_Hours = cpu_hours,
  Memory_Usage_MB = memory_usage,
  Execution_Time_s = execution_time
  )
  
  # Sauvegarder les résultats dans un fichier CSV
  write.csv(results_df, file.path(output_dir, "alra_benchmark_results.csv"), row.names = FALSE)
}

# Exemple d'utilisation
cell_counts <- c('1000','10000','50000'#, '100000'
)
num_runs <- 5
output_dir <- "/storage/Implem/pyALRA_package/pyALRA/alra_benchmark_results/R/E-MTAB-8142/"  # Spécifier votre répertoire de sortie

benchmark_alra_r("/storage/Implem/pyALRA_package/pyALRA/alra_benchmark_results/python/E-MTAB-8142/", cell_counts, num_runs, output_dir)