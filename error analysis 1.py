# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:01:54 2016

@author: aviral
"""


import scipy, numpy
import scipy.optimize, scipy.stats
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
import statsmodels.stats
import statsmodels.stats.stattools as stools

def fitdata(f, Xdata,Ydata,Errdata, pguess, ax=False, ax2=False):
    '''
    popt = vector of length N of the optimized parameters
    pcov = Covariance matrix of the fit
    perr = vector of length N of the std-dev of the optimized parameters
    p95 = half width of the 95% confidence interval for each parameter 
    p_p = vector of length N of the p-value for the parameters being zero
    (if p<0.05, null hypothesis rejected and parameter is non-zero) 
    chisquared = chisquared value for the fit
    chisquared_red = chisquared/degfreedom
    chisquare = (p, chisquared, chisquared_red, degfreedom) 
    p = Probability of finding a chisquared value at least as extreme as the one shown
    chisquared_red = chisquared/degfreedom. value should be approx. 1 for a good fit. 
    R2 = correlation coefficient or proportion of explained variance 
    R2_adj = adjusted R2 taking into account number of predictors 
    resanal = (p, w, mean, stddev)
    Analysis of residuals 
    p = Probability of finding a w at least as extreme as the one observed (should be high for good fit) 
    w = Shapiro-Wilk test criterion 
    mean = mean of residuals 
    p_res = probability that the mean value obtained is different from zero merely by chance 
    F = F-statistic for the fit msm/msE. 
    Null hypothesis is that there is NO Difference between the two variances. 
    p_F = probability that this value of F can arise by chance alone.
    p_F < 0.05 to reject null hypothesis and prove that the fit is good.
    dw = Durbin_Watson statistic (value between 0 and 4). 
    2 = no-autocorrelation. 0 = .ve autocorrelation, 4 = -ve autocorrelation. 
'''    
    
    def error(p,Xdata,Ydata,Errdata):
        Y=f(Xdata,p)
        residuals=(Y-Ydata)/Errdata
        return residuals
    res=scipy.optimize.leastsq(error,pguess,args=(Xdata,Ydata,Errdata),full_output=1)
    (popt,pcov,infodict,errmsg,ier)=res
    perr=scipy.sqrt(scipy.diag(pcov))

    M=len(Ydata)
    N=len(popt)
    #Residuals
    Y=f(Xdata,popt)
    residuals=(Y-Ydata)/Errdata
    meanY=scipy.mean(Ydata)
    squares=(Y-meanY)/Errdata
    squaresT=(Ydata-meanY)/Errdata
    
    SSM=sum(squares**2) #Corrected Sum of Squares
    SSE=sum(residuals**2) #Sum of Squares of Errors
    SST=sum(squaresT**2)#Total Corrected sum of Squares
    
    DFM=N-1 #Degree of Freedom for model
    DFE=M-N #Degree of Freedom for error
    DFT=M-1 #Degree of freedom total
    
    MSM=SSM/DFM #Mean Squares for model(explained Variance)
    MSE=SSE/DFE #Mean Squares for Error(should be small wrt MSM) unexplained Variance
    MST=SST/DFT #Mean squares for total
    
    R2=SSM/SST #proportion of unexplained variance 
    R2_adj= 1-(1-R2)*(M-1)/(M-N-1) #Adjusted R2
    
    #t-test to see if parameters are different from zero
    t_stat=popt/perr #t-stat for popt different from zero
    t_stat=t_stat.real
    p_p= 1.0-scipy.stats.t.cdf(t_stat,DFE) #should be low for good fit
    z=scipy.stats.t(M-N).ppf(0.95)
    p95=perr*z
    #Chi-Squared Analysis on Residuals
    chisquared=sum(residuals**2)
    degfreedom=M-N
    chisquared_red=chisquared/degfreedom
    p_chi2=1.0-scipy.stats.chi2.cdf(chisquared, degfreedom)
    stderr_reg=scipy.sqrt(chisquared_red)
    chisquare=(p_chi2,chisquared,chisquared_red,degfreedom,R2,R2_adj)
    
    #Analysis of Residuals
    w, p_shapiro=scipy.stats.shapiro(residuals)
    mean_res=scipy.mean(residuals)
    stddev_res=scipy.sqrt(scipy.var(residuals))
    t_res=mean_res/stddev_res #t-statistics
    p_res=1.0-scipy.stats.t.cdf(t_res,M-1)
    
    F=MSM/MSE
    p_F=1.0-scipy.stats.f.cdf(F,DFM,DFE)
    
    dw=stools.durbin_watson(residuals)
    resanal=(p_shapiro,w,mean_res,p_res,F,p_F,dw)
    
    if ax:
        formataxis(ax)
        ax.plot(Ydata,Y,'ro')
        ax.errorbar(Ydata,Y,yerr=Errdata, fmt='.')
        Ymin,Ymax=min((min(Y),min(Ydata))),max((max(Y),max(Ydata)))
        ax.plot([Ymin,Ymax],[Ymin,Ymax],'b')
        
        ax.xaxis.label.set_text('Data')
        ax.yaxis.label.set_text('Fitted')
        sigmay,avg_stddev_data=get_stderr_fit(f,Xdata,popt,pcov)
        Yplus=Y+sigmay
        Yminus=Y-sigmay
        ax.plot(Y,Yplus,'c',alpha=0.6,linestyle='--',linewidth=0.5)
        ax.plot(Y,Yminus,'c',alpha=0.6,linestyle='==',linewidth=0.5)
        ax.fill_between(Y,Yminus,Yplus,facecolor='cyan',alpha=0.5)
        titletext='Parity plot for fit.\n'
        titletext+=r'$r^2$=%5.2f,$r^2_{adj}$=%5.2f,$p_{shapiro}$=%5.2f,$Durbin-Watson=%2.1f'
        titletext+='\n F=%5.2f,$p_F$=%3.2e'
        titletext+='$\sigma_{err}^{reg}$=%5.2f'
        
        #ax.title.set_text(titletext%(R2, R2_adj, avg_stddev_data, chisquared_red, p_chi2, stderr_reg))
        ax.figure.canvas.draw()
    
    if ax2:
        formataxis(ax2)
        ax2.plot(Y,residuals,'ro')
        ax2.xaxis.label.set_text('Fitted Data')
        ax2.yaxis.label.set_text('Residuals')
        
        titletext='Analysis of Residuals\n'
        titletext+=r'mean=%5.2f,$p_{res}$=%5.2f,$p_{shapiro}$=%5.2f,$Durbin-Watson$=%2.1f'
        titletext+='\n F=%5.2f,$p_F$=%3.2e'
        ax2.title.set_text(titletext%(mean_res,p_res,p_shapiro,dw,F,p_F))
    return popt,pcov,perr, p95, p_p,chisquare, resanal
    
def get_stderr_fit(f,Xdata,popt, pcov):
    Y=f(Xdata,popt)
    listdY=[]
    for i in xrange(len(popt)):
        p=popt[i]
        dp=abs(p)/1e6 + 1e-20
        popt[i] += dp
        Yi = f(Xdata,popt)
        dY = (Yi-Y)/dp
        listdY.append(dY)
        popt[i] -= dp
    listdY=scipy.array(listdY)
    left=scipy.dot(listdY.T,pcov)
    right=scipy.dot(left,listdY)
    sigma2y=right.diagonal()
    mean_sigma2y=scipy.mean(right.diagonal())
    M=Xdata.shape[1]
    N=len(popt)
    avg_stddev_data=scipy.sqrt(M*mean_sigma2y/N)
    sigmay=scipy.sqrt(sigma2y)
    return sigmay,avg_stddev_data

def formataxis(ax):
    ax.xaxis.label.set_fontname('Georgia')
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontname('Georgia')
    ax.yaxis.label.set_fontsize(12)
    ax.title.set_fontname('Georgia')
    ax.title.set_fontsize(12)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

def import_data(xlfile,sheetname):
    df=pd.read_excel(xlfile,sheetname=sheetname)
    return df

def prepare_data(df,Criterion,Predictors,Error=False):
    Y=scipy.array(df[Criterion])
    if Error:
        Errdata=scipy.array(df[Error])
    else: 
        Errdata=scipy.ones(len(Y))
    Xdata=[]
    for X in Predictors:
        X=list(df[X])
        Xdata.append(X)
    Xdata=scipy.array(Xdata)
    return Xdata, Y, Errdata

if __name__=='__main__':
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.show
    
    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    fig2.show()
    
    def f(X,p):
        (x,y)=X
        Y=p[0]*x
        return Y
    
    df=import_data('pipefittings.xlsx','observations')
    Xdata,Ydata, Errdata=prepare_data(df,'Ydata',('x','y'),Error='err')
    #print Xdata
    #print Ydata
    #print Errdata
    
    #initial Guess
    N=1
    pguess=N*[0.0]
    
    popt,pcov,perr,p95,p_p,chisquare,resanal=fitdata(f,Xdata,Ydata,Errdata,pguess,ax=ax,ax2=ax2)
    print popt
    