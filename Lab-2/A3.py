import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
df=pd.read_excel(
    r"C:\Users\mamilla\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",
    sheet_name="IRCTC Stock Price"
)
p=df['Price'].values
m=np.mean(p)
v=np.var(p)
print("mean",m)
print("variance",v)

def mean(d):
    return sum(d)/len(d)
def var(d):
    me = mean(d)
    return sum((x-me)**2 for x in d)/len(d)
meanmy=mean(p)
varmy=var(p)
print("my cal of mean", meanmy)
print("my cal of var", varmy)

def timefunc(func,d):
    t=0
    for _ in range(10):
        s=time.time()
        func(d)
        e=time.time()
        t+=(e-s)
    return t/10

meanusingnp=timefunc(np.mean,p)
meanfunc=timefunc(mean,p)

varusingnp=timefunc(np.var,p)
varusingfunc=timefunc(var,p)
print("mean using np",meanusingnp)
print("mean with func",meanfunc)
print("var using np",varusingnp)
print("var with func",varusingfunc)


wedp=df[df['Day'] == 'Wed']['Price']
wedmean= wedp.mean()
print("mean of only wendesdays",wedmean)
print("mean of the populations",m)


df['Date']=pd.to_datetime(df['Date'])
aprice=df[df['Date'].dt.month == 4]['Price']
amean= aprice.mean()
print("mean of april",amean)
print("mean of the populations",m)

chg=df['Chg%']
probloss=(chg<0).mean()
print("P loss=",probloss)


ppwed=(df[(df['Day']=='Wed')]['Chg%'] > 0).mean()
print("probablity of prof on wed=", ppwed)


n=((df['Day']=='Wed')&(df['Chg%'] > 0)).sum()
den=(df['Day']=='Wed').sum()
conditonprob=n/den
print("conditional probability of profit and wed", conditonprob)


plt.scatter(df['Day'], df['Chg%'])
plt.show()






