#from theano import function
#import theano
#import theano.tensor as T
from scipy import stats
import numpy as np
from itertools import chain,combinations
import pandas as pd
def preprocX(Xinput):
	Xinput=np.array(Xinput)
	n,p=Xinput.shape
	if p>n:
		Xinput=Xinput.T
		n,p=Xinput.shape
	if not any(all(xval==1 for xval in Xinput[:,i]) for i in range(p)):
		Xinput = np.vstack([np.ones(n),Xinput.T]).T
		p+=1
	return n,p,Xinput
def linearRegression(Xinput,yinput,labels):
	#n = len(y)
	#p = len(X[0])+1
	#Xinput = T.matrix('Xinput')
	#yinput = T.vector('yinput')
	# n_data = T.scalar('n_data')
	n,p,Xinput=preprocX(Xinput)
	Qxx = np.dot(Xinput.T,Xinput)
	df=n-p
	preSE=np.linalg.inv(Qxx)
	beta = np.dot(np.dot(np.linalg.inv(Qxx),Xinput.T),yinput)
	hat=np.dot(Xinput,np.dot(np.linalg.inv(Qxx),Xinput.T))
	error = yinput - np.dot(Xinput,beta)
	s_squared = np.dot(error.T,error)/(n-p)
	SE = np.array([[np.sqrt(s_squared*preSE[i][i])] for i in range(p)])
	#print beta,SE
	tstats = [beta[i]/SE[i] for i in range(len(beta))]
	#print tstats
	p_values=[[2*stats.t.cdf(tstat,df)[0]] if tstat<0 else [2*stats.t.cdf(-tstat,df)[0]] for tstat in tstats]
	#return beta.tolist(),SE.T[0].tolist(),np.array(tstats).T[0].tolist(),np.array(p_values).T[0].tolist()
	r_squared = 0
	AIC = 0#2*p-2*np.log(beta)
	AICc = AIC +2*(p+1)*(p+2)/(n-p-2)
	BIC = 0#np.log(n)*p-2*np.log(beta)
	return {'coefficients':pd.DataFrame({'beta':beta,'SE':SE.T[0],'tstat':np.array(tstats).T[0],'p_val':np.array(p_values).T[0]},labels),'r_squared':r_squared,'AIC':AIC,'AICc':AICc,'BIC':BIC}

def testInputs(Xinput,y,best=False):
	n,p,Xinput=preprocX(Xinput)
	results={}
	for combo in chain.from_iterable(combinations(range(p), r) for r in range(1,p+1)):
		res=linearRegression(Xinput[:,combo],y,combo)
		if all(x<0.05 for x in res['coefficients']['p_val'].values):
			results[combo]=res
	return results

	
	
