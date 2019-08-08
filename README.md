# Batch-training
Features of steel plate surfaces were collected into the steel plate faults dataset by Semeion, Research Centre of Sciences of Communication in Italy. In this case study, the raw data comes from the steel plate faults dataset. The dataset cab be obtained from http://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults


original_data is the raw data in steel plate faults dataset.
preprocessed_data is the preprocessed data without being normalised.


# Pre_process:

Step 1. Column names are adding to the dataset.

Step 2. Effects coding is used for encoding column TypeOfSteel_A300 and column TypeOfSteel_A400.

Step 3. Delete Other_Faults class.

Step 4. Encode 6 classes into number 1 to 6.

Step 5. All data are normalised. The normalisation is executed in the code part. Thus, the preprocessed_data is the data without being normalised.

# Code

The preprocessed_data is used to train the net.

code_with_improvement.py is the code for the normal model

code_without_improvement.py is the code for the new model with Bimodal Distribution Removal
