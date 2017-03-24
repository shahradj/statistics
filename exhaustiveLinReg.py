#from theano import function
#import theano
#import theano.tensor as T
from scipy import stats
import numpy as np
from itertools import chain,combinations
import pandas as pd
def preprocX(Xinput,otherTransforms=False):
	Xinput=np.array(Xinput)
	n,p=Xinput.shape
	if p>n:
		Xinput=Xinput.T
		n,p=Xinput.shape
	if not any(all(xval==1 for xval in Xinput[:,i]) for i in range(p)):
		Xinput = np.vstack([np.ones(n),Xinput.T]).T
		p+=1
	if otherTransforms:
		tinput=[Xinput.T]
		tinput+=[np.sqrt(Xinput[:,i]) for i in range(p) if not all(xval==1 for xval in Xinput[:,i])]
		tinput+=[np.log(Xinput[:,i]) for i in range(p) if not all(xval==1 for xval in Xinput[:,i])]
		tinput+=[np.exp(Xinput[:,i]) for i in range(p) if not all(xval==1 for xval in Xinput[:,i])]
		tinput+=[Xinput[:,i]**2 for i in range(p) if not all(xval==1 for xval in Xinput[:,i])]
		Xinput=np.vstack(tinput).T
		n,p=Xinput.shape
	return n,p,Xinput


	ROI = a + b*impressions +c*clicks
def linearRegression(Xinput,yinput,labels):
	#n = len(y)
	#p = len(X[0])+1
	#Xinput = T.matrix('Xinput')
	#yinput = T.vector('yinput')
	# n_data = T.scalar('n_data')
	n,p=Xinput.shape
	yinput=np.array(yinput)
	Qxx = np.dot(Xinput.T,Xinput)
	df=n-p
	preSE=np.linalg.inv(Qxx)
	beta = np.dot(np.dot(preSE,Xinput.T),yinput)
	hat=np.dot(Xinput,np.dot(preSE,Xinput.T))
	error = yinput - np.dot(Xinput,beta)
	ymean=np.mean(yinput)
	total = yinput - np.repeat(ymean,n)#array([[ymean] for i in range(n)])
	#return total,error
	s_squared = np.dot(error.T,error)/(n-p)
	SE = np.array([[np.sqrt(s_squared*preSE[i][i])] for i in range(p)])
	L=np.diag(np.ones(n))-np.dot([[x] for x in np.ones(n)],[[x for x in np.ones(n)]])
	M=np.diag(np.ones(n))-hat
	#print beta,SE
	tstats = [beta[i]/SE[i] for i in range(len(beta))]
	#print tstats
	p_values=[[2*stats.t.cdf(tstat,df)[0]] if tstat<0 else [2*stats.t.cdf(-tstat,df)[0]] for tstat in tstats]
	#return beta.tolist(),SE.T[0].tolist(),np.array(tstats).T[0].tolist(),np.array(p_values).T[0].tolist()
	SSE=np.dot(error.T,error)
	SST=np.dot(total.T,total)

	TSS=np.dot(yinput.T,np.dot(L,yinput))

	r_squared = 1-SSE/SST
	adj_r_sq=1-(n-1)/(n-p)*SSE/SST
	#(s_squared*(n-p)/(TSS/(n-1)))
	#r_squared = (1-SSR/(n)/(TSS/(n)))
	
	AIC = 0#2*p-2*np.log(beta)
	AICc = AIC +2*(p+1)*(p+2)/(n-p-2)
	BIC = 0#np.log(n)*p-2*np.log(beta)
	return {'coefficients':pd.DataFrame({'beta':beta,'SE':SE.T[0],'tstat':np.array(tstats).T[0],'p_val':np.array(p_values).T[0]},labels),'adj_r_squared':adj_r_sq,'r_squared':r_squared,'AIC':AIC,'AICc':AICc,'BIC':BIC}

def testInputs(df,inputVars,outputVar,onlyBest=False,otherTransforms=False):
	n,p,Xinput=preprocX(df[inputVars].values,otherTransforms)
	cols=['Intercept']+inputVars
	y=df[outputVar].values
	if otherTransforms:
		cols+=['sqrt_'+x for x in inputVars]
		cols+=['log_'+x for x in inputVars]
		cols+=['exp_'+x for x in inputVars]
		cols+=['squar_'+x for x in inputVars]
	results={}
	for combo in chain.from_iterable(combinations(range(p), r) for r in range(1,p+1)):
		colNames=[cols[i] for i in combo]
		res=linearRegression(Xinput[:,combo],y,colNames)
		if onlyBest:
			if all(x<0.05 for x in res['coefficients']['p_val'].values) and all(x>1e-6 for x in res['coefficients']['beta'].values):
				results[combo]=res
		else:
			results[combo]=res
	return results
def getData():
	ac=AdwordsClient('/home/alasdair/Documents/NelsonBack/sparroads.yaml','car rental','G | Rental Car | Generic (Exact)','')

	selector={'fields':['Id','Impressions','Conversions','FirstPositionCpc','Clicks','Cost','Date','AverageCpc','DayOfWeek'],'predicates':[{'field':'Id','operator':'EQUALS','values':[ac.keywordId]}]}
	reportParams={'reportType':'CRITERIA_PERFORMANCE_REPORT','dateRangeType':'ALL_TIME','selector':selector,'reportName':'report for me','downloadFormat':'CSV'}
	dl=ac.client.GetReportDownloader('v201609')
	rawreport=dl.DownloadReportAsString(reportParams,skip_report_header=True,skip_column_header=False,skip_report_summary=True)
	report=pd.DataFrame.from_csv(StringIO(rawreport),sep=',').sort_values('Day')
	costcols=['Cost','First position CPC','Avg. CPC']
	for col in costcols:
		report.loc[:,col.lower()]=report[col]*1e-6
	report.loc[:,'ConvShift']=report['Conversions'].shift(-1)
	report.loc[:,'ClickShift']=report['Clicks'].shift(-1)
	weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
	report.loc[:,'weekday']=[weekdays.index(x) for x in report['Day of week']]
	for wday in report['Day of week'].drop_duplicates().values:
		report.loc[:,wday]=[(x==wday)*1 for x in report['Day of week']]
	return report
