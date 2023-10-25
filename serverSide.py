#Data Source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import pickle
import pandas as pd
import numpy as np
import time
import phe as paillier
import json
from Pyfhel import Pyfhel, PyCtxt, PyPtxt

class PreProcessor:
  def __init__(self):
    self.trainColumns = ['LotArea','OverallQual',
                   'OverallCond','TotalBsmtSF','GrLivArea','TotRmsAbvGrd',
                   'GarageArea','SalePrice']
    self.testColumns = ['LotArea','OverallQual',
                   'OverallCond','TotalBsmtSF','GrLivArea','TotRmsAbvGrd',
                   'GarageArea']
  
  def featureExtraction(self,df,cols,outFileName):
    numeric = pd.DataFrame()
    for col in cols:
      numeric[col] = df[col]
    numeric.to_csv(outFileName)
    return numeric
  
  def get_mean(self,x):
    return sum(x)/len(x)

  def get_std(self,x, mean):
    return ( (x-mean).dot(x-mean) / len(x) )**0.5

  def get_mean_and_std(self,x):
    mean = self.get_mean(x)
    std = self.get_std(x, mean)
    return mean, std

  def normalization(self,x, mean, std):
    return ( x - mean )/std

  def proc_train_data(self,datapath):
    """
    standardize the training data and output it to an external file 'train_normalized.txt'
    """
    
    # find the means and stds for all 8 relevant columns
    # LotArea	OverallQual	OverallCond	TotalBsmtSF	GrLivArea	TotRmsAbvGrd	GarageArea	SalePrice
    original = pd.read_csv(datapath,header=0,index_col=0)
    housing = self.featureExtraction(original,self.trainColumns,'train_numeric.csv')
    lot_mean, lot_std = self.get_mean_and_std(housing['LotArea'])
    qual_mean, qual_std = self.get_mean_and_std(housing['OverallQual'])
    cond_mean, cond_std = self.get_mean_and_std(housing['OverallCond'])
    bsmt_mean, bsmt_std = self.get_mean_and_std(housing['TotalBsmtSF'])
    liv_mean, liv_std = self.get_mean_and_std(housing['GrLivArea'])
    rms_mean, rms_std = self.get_mean_and_std(housing['TotRmsAbvGrd'])
    garage_mean, garage_std = self.get_mean_and_std(housing['GarageArea'])
    price_mean, price_std = self.get_mean_and_std(housing['SalePrice'])

    # standardize the training data
    norm_lot = self.normalization(housing['LotArea'],lot_mean,lot_std)
    norm_qual = self.normalization(housing['OverallQual'],qual_mean,qual_std)
    norm_cond = self.normalization(housing['OverallCond'],cond_mean,cond_std)
    norm_bsmt = self.normalization(housing['TotalBsmtSF'],bsmt_mean,bsmt_std)
    norm_liv = self.normalization(housing['GrLivArea'],liv_mean,liv_std)
    norm_rms = self.normalization(housing['TotRmsAbvGrd'],rms_mean,rms_std)
    norm_garage = self.normalization(housing['GarageArea'],garage_mean,garage_std)
    norm_price = self.normalization(housing['SalePrice'],price_mean,price_std)

    # save the normalized data to 'training_normalized.txt'
    norm = pd.DataFrame(zip(norm_lot,norm_qual,norm_cond,norm_bsmt,norm_liv,norm_rms,norm_garage,norm_price))
    norm.columns = self.trainColumns
    norm.to_csv("train_normalized.csv")

    # store the values used for normalization
    f = open("mean_std.pk", "wb")
    pickle.dump({'LotArea':{'mean': lot_mean, 'std': lot_std},
                 'OverallQual':{'mean': qual_mean, 'std': qual_std},
                 'OverallCond':{'mean': cond_mean, 'std': cond_std},
                 'TotalBsmtSF':{'mean': bsmt_mean, 'std': bsmt_std},
                 'GrLivArea':{'mean': liv_mean, 'std': liv_std},
                 'TotRmsAbvGrd':{'mean': rms_mean, 'std': rms_std},
                 'GarageArea':{'mean': garage_mean, 'std': garage_std},
                 'SalePrice':{'mean': price_mean, 'std': price_std},
                 },
                f)
    f.close()
    
    return

  def proc_test_data(self,datapath):
    """
    standardize the testing data and output it to an external file 'test_normalized.txt'
    """
    
    # Extract the 7 relevant columns form testing data
    # LotArea	OverallQual	OverallCond	TotalBsmtSF	GrLivArea	TotRmsAbvGrd	GarageArea	
    original = pd.read_csv(datapath,header=0,index_col=0)
    housing = self.featureExtraction(original,self.testColumns,'test_numeric.csv')
    
    return

from sklearn.linear_model import LinearRegression

class LinearRegressionModel:
  def __init__(self):
    self.model = LinearRegression()
    self.pp = PreProcessor()
  
  def train(self):
    self.pp.proc_train_data("train.csv")
    df = pd.read_csv("train_normalized.csv",index_col=0)
    y = df['SalePrice']
    x = df.drop(['SalePrice'],axis=1)
    self.model.fit(x.values,y.values)
  
  def getCoef(self):
    return self.model.coef_
  
  
  def predict(self,df):

    # get the values used for normalization
    f = open('mean_std.pk', 'rb')
    norm_params = pickle.load(f)
    lot_mean, lot_std = norm_params['LotArea']["mean"],norm_params['LotArea']["std"]
    qual_mean, qual_std = norm_params['OverallQual']["mean"],norm_params['OverallQual']["std"]
    cond_mean, cond_std = norm_params['OverallCond']["mean"],norm_params['OverallCond']["std"]
    bsmt_mean, bsmt_std = norm_params['TotalBsmtSF']["mean"],norm_params['TotalBsmtSF']["std"]
    liv_mean, liv_std = norm_params['GrLivArea']["mean"],norm_params['GrLivArea']["std"]
    rms_mean, rms_std = norm_params['TotRmsAbvGrd']["mean"],norm_params['TotRmsAbvGrd']["std"]
    garage_mean, garage_std = norm_params['GarageArea']["mean"],norm_params['GarageArea']["std"]
    price_mean, price_std = norm_params['SalePrice']["mean"],norm_params['SalePrice']["std"]

    f.close()

    # standardize the input data for prediction
    norm_lot = self.pp.normalization(df['LotArea'],lot_mean,lot_std)
    norm_qual = self.pp.normalization(df['OverallQual'],qual_mean,qual_std)
    norm_cond = self.pp.normalization(df['OverallCond'],cond_mean,cond_std)
    norm_bsmt = self.pp.normalization(df['TotalBsmtSF'],bsmt_mean,bsmt_std)
    norm_liv = self.pp.normalization(df['GrLivArea'],liv_mean,liv_std)
    norm_rms = self.pp.normalization(df['TotRmsAbvGrd'],rms_mean,rms_std)
    norm_garage = self.pp.normalization(df['GarageArea'],garage_mean,garage_std)

    norm_input = pd.DataFrame(zip(norm_lot,norm_qual,norm_cond,norm_bsmt,norm_liv,norm_rms,norm_garage))
    pred_norm_price = self.model.predict(norm_input.values)

    return pred_norm_price*price_std + price_mean
  
  #make predictions on plaintext input and store result in "plain_text_pred.csv"
  def plainText(self,dp):
    self.pp.proc_test_data("test.csv")
    to_pred=pd.read_csv(dp,index_col=0,header=0)
    pred = self.predict(to_pred)
    np.savetxt("plain_text_pred.csv", pred, fmt="%.8f",delimiter=" ")
    #print("Predicted price(s) is:",pred)

def getInputPaillier():
  with open('data.json', 'r') as file:
    d = json.load(file)
    data = json.loads(d)
    return data

def computePaillier(price_mean,price_std, lr,mean,std):
  data = getInputPaillier()
  mycoef = lr.getCoef()
  pk = data['public key']
  pub_key = paillier.PaillierPublicKey(n = int(pk['n']))
  enc_nums_rec = [paillier.EncryptedNumber(pub_key, int(x[0], int(x[1]))) for x in data['values']]
  result = sum([mycoef[i] * normalize(enc_nums_rec[i],mean[i],std[i]) for i in range(len(mycoef))])
  result = result*price_std
  result = result + price_mean
  return result, pub_key

def Paillier(price_mean,price_std, lr, mean,std):
    results, pub_key = computePaillier(price_mean,price_std, lr,mean,std)
    encrypted_data = {}
    encrypted_data['pub_key'] = {'n': pub_key.n}
    encrypted_data['values'] = (str(results.ciphertext()), results. exponent)
    datafile = json.dumps(encrypted_data)
    with open('answer.json', 'w') as file:
      json.dump(datafile, file)

def computeFHE(num, price_mean,price_std, mean, std):
  data = pickle.load(open("cipher.p", "rb"))
  print("Data transmitted from client side: \n")
  print(data)
  print("\n")
  #instances imported, secret key excluded
  HE = data["he"]
  del data["he"]
  #Initialize the instances of the pyfhel ciphertext objects
  for k,v in data.items():
    v._pyfhel = HE
  res = data["res"]
  del data["res"]
  i = 0
  for k,v in data.items():
    v = normalize(v,mean[i],std[i])
    mul = v * num[i]
    res = res + mul
    i+=1
  r = res*price_std + price_mean
  #store the result in a file to return back to the client side
  output = dict()
  output["res"] = r
  output["he"] = HE
  pickle.dump(output, open("res.p", "wb"))
    
def normalize(num, mean, std):
  return (num-mean)/std
    
def main():
    lr = LinearRegressionModel()
    lr.train()
    lr.getCoef()
    f = open('mean_std.pk', 'rb')
    norm_params = pickle.load(f)
    lot_mean, lot_std = norm_params['LotArea']["mean"],norm_params['LotArea']["std"]
    qual_mean, qual_std = norm_params['OverallQual']["mean"],norm_params['OverallQual']["std"]
    cond_mean, cond_std = norm_params['OverallCond']["mean"],norm_params['OverallCond']["std"]
    bsmt_mean, bsmt_std = norm_params['TotalBsmtSF']["mean"],norm_params['TotalBsmtSF']["std"]
    liv_mean, liv_std = norm_params['GrLivArea']["mean"],norm_params['GrLivArea']["std"]
    rms_mean, rms_std = norm_params['TotRmsAbvGrd']["mean"],norm_params['TotRmsAbvGrd']["std"]
    garage_mean, garage_std = norm_params['GarageArea']["mean"],norm_params['GarageArea']["std"]
    price_mean, price_std = norm_params['SalePrice']["mean"],norm_params['SalePrice']["std"]
    mean = [lot_mean, qual_mean, cond_mean, bsmt_mean, liv_mean, rms_mean, garage_mean]
    std = [lot_std, qual_std, cond_std, bsmt_std, liv_std, rms_std, garage_std]
    f.close()
    f = open('test1.csv', 'r')
    results = f.readlines()
    plainCalc = results[0]
    f.close()
    
    mode = int(input("Modes:\n1:Paillier Computation\n2:FHE computation\nSelect mode(1/2):"))
    if mode ==1:    
      print("-------Paillier computation starts--------")
      start_time = time.time()
      Paillier(price_mean,price_std, lr,mean,std)
      end_time = time.time()
      print(" Pallier takes " + str(end_time-start_time) + " seconds to compute the result.\n")
    elif mode ==2:
      print("-------FHE computation starts--------")
      start_time = time.time()
      computeFHE(lr.getCoef(),price_mean,price_std, mean,std)
      end_time = time.time()
      print(" FHE takes " + str(end_time-start_time) + " seconds to compute the result.\n")
    else:
      print("Mode not defined!")
    
    
main()
