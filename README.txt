Project description:

Application of dimensionality reduction algorithms like Principal Component Analysis (PCA), Independent Component Analysis (ICA), Random Projection (RP), Random Forest
(RF) and Linear Discriminant Analysis (LDA) on Bioresponse and Letter Recognition datasets, followed by application of unsupervised clustering techniques like k-Means clustering and GMM (Gaussian Mixture Models - based on Expectation Maximization) on both the datasets.

The following methodology and steps were used for both datasets in Python 3.6 and Windows 10 machine.

Scikit-learn was used for implementations and the datasets used are letter recognition and bioresponse.

Project_code has the following python files:
1. Clustering_step1.py that clusters letter recognition dataset using k-means and GMM (EM)
2. 2_Clustering_step1.py that clusters bioresponse dataset using k-means and GMM (EM)
3. PCA.py that applies Principal Component Analysis to letter recognition dataset and then inputs the PCA-applied data to NN(Neural Network) from Assignment-1.
4. 2_PCA.py that applies Principal Component Analysis to  bioresponse dataset and plots and records the results in PCA folder.
5. ICA.py that applies Independent Component Analysis to letter recognition dataset and then inputs the ICA-applied data to NN from Assignment-1.
6. 2_ICA.py that applies Independent Component Analysis to  bioresponse dataset and plots and records the results in ICA folder.
7. RP.py that applies Random projection to letter recognition dataset and then inputs the RP-applied data to NN from Assignment-1.
8. 2_RP.py that applies Random Projection to  bioresponse dataset and plots and records the results in RP folder.
9. RF.py that applies Random Forest to letter recognition dataset and then inputs the RF-applied data to NN from Assignment-1, plots and records the results in RF folder and apply RF to bioresponse dataset.
10. LDA.py that applies Linear Discriminant Analysis to  letter recognition dataset and then inputs the LDA-applied data to NN from Assignment-1
11.2_LDA.py that applies Linear Discriminant Analysis to bioresponse dataset.
12.Clustering_step5.py takes as parameter, one of the above dimensionality reduction(DR) algorithm and then the data reduced by that DR is clusterd using k-means and GMM, for letter recognition dataset and then inputted to NN.
13.TSNE.py computes the data required to plot the t-SNE plot for non DR-applied initial data. It takes as parameter "TSNE" so as to output the results to TSNE folder.
14.plotting.py does the plotting of t-SNE for all DR-applied data.
15.get_data.py gets the letter recognition dataset from openML and store it locally as d.pkl file.
16.get_data_next_dataset.py gets the bioresponse dataset from openML and store it locally as bioresponse.pkl file.
17.helpers.py has the helping functions being called and used by several python files in the project.

Other files inside the parent folder are:
1. d.pkl stores the letter recognition data in .pkl format
2. bioresponse.pkl stores the bioresponse dataset in .pkl format.
3. letter_recognition.csv has the letter recognition dataset from OpenML
4. Bioresponse.csv has the bioresponse dataset from OpenML

The folders included in the parent folder contain the results of the various experiments:
1.Output_letter folder has the grid search results of NN trained using data not preprocessed by DR and the results of experiments used to find optimal k for k-means and GMM. clustering_step1.py and 2_clustering_step1.py outputs results into this folder. It has only the results for letter recognition dataset.
2. Output_Bioresponse folder has the results of experiments used to find optimal k for k-means and GMM for bioresponse dataset. clustering_step1.py and 2_clustering_step1.py outputs into this folder.
3.TSNE folder has data used to construct t-SNE plots for initial non-DR-applied data. TSNE.py outputs into this folder.
4.PCA folder has the results of gridsearch on NN, and clustering for both datasets for PCA-applied data. PCA.py, 2_PCA.py and clustering_step5.py outputs results into this folder.
5.ICA folder has the results of gridsearch on NN, and clustering for both datasets for ICA-applied data. ICA.py, 2_ICA.py and clustering_step5.py outputs results into this folder.
6.RP folder has the results of gridsearch on NN, and clustering for both datasets for RP applied data. RP.py, 2_RP.py and clustering_step5.py outputs results into this folder.
7.RF folder has the results of gridsearch on NN, and clustering for both datasets for RF applied data. RF.py, 2_RF.py and clustering_step5.py outputs results into this folder.

Detailed steps and instructions to run the code:

The steps are:
1. Run the clustering algorithms on the datasets 
2. Apply the dimensionality reduction algorithms to the two datasets 
3. Reproduce clustering experiments, but on the data after dimensionality reduction is run on it.
4. Apply the dimensionality reduction algorithms to letter recognition dataset 
5. Apply the clustering algorithms to the same dataset, treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun neural network learner on the newly projected data.

NB: Plots are not programmed to generate all-together in any file, so as to avoid confusion with many plots and for a more organised experiment. 

Step 1 is done for letter recognition dataset by clustering_step1.py. 
It can be run on python IDE or command prompt and it will generate plots for finding optimal k. The plots are included in the report. When a generated plot is closed, the program resumes to generate the next plot. 
Step1 is done for bioresponse dataset using 2_clustering_step1.py and it generates the relevant plot one-by-one. Next plot appears when the previous plot is closed.

Step 2 is done by PCA.py, ICA.py, RP.py and RF.py for letter recognition dataset and 2_PCA.py, 2_ICA.py, 2_RP.py and RF.py for bioresponse_dataset. LDA.py and 2_LDA.py are also present, however, they are not part of final project. I used LDA first, but since it did not work well for bioresponse dataset as it has binary class, I used RF as my DR of choice.

Step 3 is done by clustering_step2.py for letter recognition dataset and 2_clustering_step2.py for bioresponse dataset. The parameter can be set as PCA, ICA ,RP or RF for using the corresponding DR-applied data.

Step 4 is already performed by Step2 as letter recognition dataset is the one from assignment-1.

Step-5 is done by clustering_step5.py for letter recognition dataset. The parameter can be set as PCA, ICA ,RP or RF for using the corresponding DR-applied data. [Bioresponse dataset does not need NN experiments]

In a nutshell, all files except clustering_step2.py, 2_clustering_step2.py, clustering_step5.py, TSNE can be run without any parameters (just by clicking the run button on IDE) and they will generate the relevant graphs and statistics one by one. The files needing parameters (above mentioned) can be given a parameter among PCA, ICA, RP or RF. TSNE.py needs the output folder name as parameter, which is "TSNE" in this case.

The datasets are present in pickle file format within the parent folder. They are also present separately as letter_recognition.csv and bioresponse.csv within the parent folder. They can also be accessed at:
1. Letter Recognition dataset: https://www.openml.org/d/6
2. Bioresponse dataset: https://www.openml.org/d/4134 

The recommended order of running the python files:

For letter recognition dataset
1. clustering_step1.py
2. PCA.py
3. clustering_step2.py
4. clustering_step5.py
5. ICA.py
6. clustering_step2.py
7. clustering_step5.py
8. RP.py
9. clustering_step2.py
10.clustering_step5.py
11.RF.py
12.clustering_step2.py
13.clustering_step5.py

For bioresponse dataset:
1. 2_clustering_step1.py
2. 2_PCA.py
3. 2_clustering_step2.py
4. 2_ICA.py
5. 2_clustering_step2.py
6. 2_RP.py
7. 2_clustering_step2.py
8. RF.py
9. 2_clustering_step2.py

Notes:

1. Since Github has a file size restriction of 100 MB, I could not upload the intermediate datsets generated by various DR algorithms. So, to run the experiment, please run the individual DR file (Eg: PCA.py) before running the clustering_step5.py file. If possible, kindly follow the above order of running python files for both datasets.

2. The grid search results of NN from clustering_step5.py and from other DR files are stored inside the corresponding DR folder."Letters_gridsearchCV.csv" has the output of NN when the DR-applied data was directly fed to NN. "Letter_cluster_<clustering algorithm>.csv" files contain the results when cluster was used as the only feature and the specified clustering algorithm was used to cluster the specifed DR-applied data."NN_with_clusterfeature_and_dim_red_features_<clustering algorithm>.csv" contains the results when cluster was used as an additional feature with other existing features of the dataset.

3. The Output_letter folder has NN gridSearch results in "Letter_cluster_<clustering algorithm>.csv" files. It contains the results when cluster was used as the only feature and the specified clustering algorithm was used to cluster the initial non DR-applied data."NN_with_clusterfeature_and_dim_red_features_<clustering algorithm>.csv" contains the results when cluster was used as an additional feature with other existing features of the dataset.

