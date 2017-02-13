from theano import function
import theano
import theano.tensor as T
import numpy as np
from itertools import chain,combinations
def linearRegression(Xinput,yinput):
	#n = len(y)
	#p = len(X[0])+1
	#Xinput = T.matrix('Xinput')
	#yinput = T.vector('yinput')
	n,p=Xinput.shape
	n_data = T.scalar('n_data')
	Qxx = np.dot(Xinput.T,Xinput)
	preSE=np.linalg.inv(Qxx/n_data)
	beta = np.dot(np.dot(np.linalg.inv(Qxx),Xinput.T),yinput)
	error = yinput - np.dot(Xinput,beta)
	s_squared = np.dot(error.T,error)/n_data
	SE = np.array([np.sqrt(s_squared*preSE[i][i]) for i in range(p)])
	tstats = beta/SE
	p_values = [np.ttest(tstat) for tstat in tstats]
	r_squared = 0
	AIC = 0#2*p-2*np.log(beta)
	AICc = AIC +2*(p+1)*(p+2)/(n-p-2)
	BIC = 0#np.log(n)*p-2*np.log(beta)
	return {'coefficients':pd.DataFrame({'beta':beta,'SE':SE,'tstat':tstats,'p_val':p_values}),'r_squared':r_squared,'AIC':AIC,'AICc':AICc,'BIC':BIC}
def testInputs(X,y,best=False):
	Xinput = np.vstack(np.ones(n),X.T).T
	n,p=Xinput.shape
	results={}
	for combo in chain.from_iterable(combinations(range(p), r) for r in range(1,p+1)):
		res=linearRegression(Xinput[:,combo],y)
		if all(x<0.05 for x in res['coefficients']['p_val'].values):
			results[combo]=res
	return results

	
	