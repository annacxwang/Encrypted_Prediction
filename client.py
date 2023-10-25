import phe as paillier
import json
from Pyfhel import Pyfhel, PyCtxt, PyPtxt
import pickle
import time

#take input from user
def getInput():
    data = dict()
    data["lotArea"] = float(input("What is the Lot size in square feet: "))
    data["OverallQual"] = float(input("Rate the overall material and finish of the house (1:very poor - 10:very excellent): "))
    data["OverallCond"] = float(input("Rate the overall condition of the house (1:very poor - 10:very excellent): "))
    data["TotalBsmtSF"] = float(input("Total square feet of basement area: "))
    data["GrLivArea"] = float(input("Above grade (ground) living area square feet: "))
    data["TotRmsAbvGrd"] = float(input("Total rooms above grade (does not include bathrooms): "))
    data["GarageArea"] = float(input("Size of garage in square feet: "))
    return data

#Generating public key and private key for Paillier using Paillier build in fuction
def storeKeys():
    public_key, private_key = paillier.generate_paillier_keypair()
    keys = {}
    keys['public_key'] = {'n': public_key.n}
    keys['private_key'] = {'p': private_key.p, 'q': private_key.q}
    with open('custkeys.json', 'w') as file:
        json.dump(keys, file)

    
def getKeys():
    with open('custkeys.json', 'r') as file:
        keys = json.load(file)
        pub_key = paillier.PaillierPublicKey(n = int(keys['public_key']['n']))
        priv_key = paillier.PaillierPrivateKey(pub_key, keys['private_key']['p'], keys['private_key']['q'])
        return pub_key, priv_key

#Encryption using Paillier
def Paillier_encrypt(userInput):
    pub_key, priv_key = getKeys()
    temp = []
    for k,val in userInput.items():
        temp.append(int(val))
    data = lotArea, OverallQual, OverallCond, TotalBsmtSF, GrLivArea, TotRmsAbvGrd, GarageArea = temp

    #encrypt the data and generate a json file to send to the company
    encrypted_data_list = [pub_key.encrypt(x) for x in data]
    encrypted_data = {}
    encrypted_data['public key'] = {'n': pub_key.n}
    encrypted_data['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_data_list]
    datafile = json.dumps(encrypted_data)
    with open('data.json', 'w') as file:
        json.dump(datafile, file)

#Load the answer file after receiving the encrypted result from the company
def loadAns():
    with open('answer.json', 'r') as file:
        answer = json.load(file)
    ans = json.loads(answer)
    return ans

#Decrypt the data print it for customers
def Paillier_decrypt():
    pub_key, priv_key = getKeys()
    answer_file = loadAns()
    answer_key = paillier.PaillierPublicKey(n = int(answer_file['pub_key']['n']))
    answer = paillier.EncryptedNumber(answer_key, int(answer_file['values'][0]), int(answer_file['values'][1]))

    #only decrypt when the public key in answer match with our public key - to verify this is the expecting result
    if(answer_key == pub_key):
        print(priv_key.decrypt(answer))


def FHE_mian():
    he = Pyfhel()
    he.contextGen(p=65537, m=4096)
    he.keyGen()

    he.savepublicKey('pub.key')
    he.savesecretKey('secret.key')
    return he


#Encryption scheme for FHE
def Enc(data, HE):
    f = open("cipher.p", "wb")
    data["res"] = float(0)
    for k,v in data.items():
        ptxt = HE.encodeFrac(v)
        ctxt = HE.encryptPtxt(ptxt)
        data[k] = ctxt
    data["he"] = HE
    pickle.dump(data,f)
    f.close()

#Decryption for FHE

def FHE_decrypt():
    data = pickle.load(open("res.p", "rb"))
    res = data["res"]
    HE = data["he"]
    HE.restorepublicKey('pub.key')
    HE.restoresecretKey('secret.key')
    r = HE.decryptFrac(res)
    print("The result is " + str(r) + "\n")

def main():
    mode = int(input("Modes:\n1:Paillier Encrypt\n2:Paillier Decrypt\n3:FHE Encrypt\n4:FHE Decrept\nSelect mode(1/2/3/4):"))
    if mode ==1:
        userInput = getInput()
        print("\n-------- Pallier encrytion starts --------")
        start_time = time.time()
        storeKeys()
        Paillier_encrypt(userInput)
        end_time = time.time()
        print("\n Paillier takes " + str(end_time-start_time) + " seconds to encrypt.\n")
    elif mode ==2:
        print("\n-------- Pallier decrytion starts --------")
        start_time = time.time()
        Paillier_decrypt()
        end_time = time.time()
        print("\n Paillier takes " + str(end_time-start_time) + " seconds to decrypt.\n")
    elif mode == 3:
        userInput = getInput()
        print("\n--------FHE encrytion starts --------")
        start_time = time.time()
        HE = FHE_mian()
        Enc(userInput, HE)
        end_time = time.time()
        print("\n FHE takes " + str(end_time-start_time) + " seconds to encrypt.\n")
    elif mode == 4:
        print("\n-------- FHE decrytion starts --------")
        start_time = time.time()
        FHE_decrypt()
        end_time = time.time()
        print("\n FHE takes " + str(end_time-start_time) + " seconds to decrypt.\n")
    else:
        print("\nMode not defined!\n")

main()
