README

About the Project

Input information about your real estate asset and we will help you encrypt your data, with the help of a third party server, we will help you calculate the predicted value of your asset. Your data will be encrypted and kept hidden from the third party server. 

Authors: Vera Li, Chunxiao Wang, Alice Yin@NYU for CS6903 Intro to Cryptography

Package Dependencies:
- Pyfhel
- Phe
- Json
- Pandas
- Numpy
- Pickle

Encryption-Specific Package installation:
- pip3 install phe
- pip3 install pyfhel

Execution roadmap

1. run client.py and choose mode 1 to encrypt data with Paillier and mode 3 with FHE 
2. Input the data as prompted
3. Wait for the encryption output file to be generated
4. run serverSide.py and choose the encrypted computation scheme (mode 1 for Paillier and 2 for FHE) accordingly
5. Wait for the encrypted computation result to be generated
6. run client.py again to decrypt the result (mode 2 with Paillier and mode 4 with FHE) generated on the server side

Initial input files:
train.csv: contains initial training data to train the machine learning model on server side 
test.csv,test1.csv: contains data used for testing the model 

Intermediate Files and Content:
custkeys.json:stores keys generated on the customer side
Data.json:stores input data encrypted by Pailler
Answer.json:stores computed data encrypted by Pailler
cipher.p:stores input data encrypted by FHE
res.p:stores computed data encrypted by FHE
mean_std.pk:stores means and stds for training dataset
Train_numeric.csv:stores the numeric features used in model training
Train_normalized.csv:stores the normalized training data set

