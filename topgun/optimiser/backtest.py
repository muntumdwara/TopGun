# -*- coding: utf-8 -*-
"""
TopGun Backtest Class

@author: David McNay & 

Adjustment made by Muntu
"""

# %% IMPORTs CELL

# Default Imports
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import kurtosis, skew, norm

# Rolling Regression
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Plotly for charting
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# %% CLASS MODULE

class BacktestAnalytics(object):
    """ Backtest Analytics & Reporting for Timeseries Data
    
    Here we take one or more portfolios (strategies) timeseries returns and 
    run various tests against a specified becnhmark timeseries. There is
    an option to provide multiple benchmark returns in the bmkrtns dataframe; 
    at the moment only one benchmark is used as a direct comparitor but those
    other indices will be used as part of a correlation analysis.
    
    NB/ THIS DOES NOT MANAGE STOCK LEVEL ANALYSIS - ONLY TIMESERIES
    
    INPUTS:
        portrtns: pd.DataFrame (or pd.Series) of timeseries returns or prices; 
            if using prices must add ports_as_rtns=False
        bmkrtns: same conditions as portrtns (& using bmks_as_rtns)
        benchmark (OPTIONAL): str() as column name in bmkrtns dataframe. If
            not provided a vector of zeros is used as the benchmark returns
        eom: True(default)|False will converts all input dates to end-of-month
        freq: 12(default) is the periods per annum. Currently only works monthly
        
    MAIN FUNCTIONS:
        run_backtest():
            * builds data from portrtns, bmkrtns & benchmark
            * creates full period summary table
            * builds and stores rolling vol, TE, IR, etc..
            * builds & saves drawdown series and analyses individual drawdowns
            * creates wider correlation matrix inc. xs_rtns
        
        plot_master():
            * creates dictionary of useful plots
            
        pretty_panda():
            * applies basic styling to a pandas table - this could move
            * bespoke sub funcs extend this; these funcs start "pretty_panda_xxx"
            
    REPORTING:
    In all cases we produce a templated markdown script with Plotly plots
    already embedded as HTML - these can be fed to report_writer or anything
    which turns markdown to a static html/pdf.
        - markdown_doc() is primary function for generating markdown. REQUIRES
            plot_master() to have been run but will prettify dataframe itself.
        
    DEVELOPMENT:
        - dynamic plots for correlation wrt time
        - more work on hit-rates
        - PCA based analysis
        - basic checking that input benchmarks or Rf in bmkrtns columns

    Author: David J McNay
    """

    def __init__(self, portrtns, bmkrtns, 
                 benchmark=None, Rf=None,
                 eom = True, freq=12,
                 ports_as_rtns=True, bmks_as_rtns=True):
        
        # ingest portfolio (strategy) and benchmark returns
        # check if supplied data is as returns or prices
        # if prices convert to returns
        self.portrtns = portrtns if ports_as_rtns else portrtns.pct_change()
        self.bmkrtns = bmkrtns if bmks_as_rtns else bmkrtns.pct_change()
        
        # convert to end-of-month dates if required
        # if we do this at the initialisation stage we know all else is eom
        if eom:
            self.portrtns = self._eom(self.portrtns)
            self.bmkrtns = self._eom(self.bmkrtns)
        
        # Name of benchmark - should match column name in bmkrtns
        # Similarly the "Risk-Free" component if being provided
        self.Rb = benchmark
        self.Rf = Rf
        
        # Other options
        self.freq = freq    # Assume 12 for monthly
        
        # Other setup things
        self.rolling = dict()    # blank dictionary for rolling window frames
        
        # Plotly template
        colourmap = ['black', 'teal', 'purple', 'grey', 'deeppink', 'skyblue', 'lime', 'green','darkorange', 'gold', 'navy', 'darkred',]
        fig = go.Figure(layout=dict(
                      font={'family':'Garamond', 'size':14},
                      plot_bgcolor= 'white',
                      colorway=colourmap,
                      showlegend=True,
                      legend={'orientation':'v'},
                      margin = {'l':75, 'r':50, 'b':25, 't':50},
                      xaxis= {'anchor': 'y1', 'title': '', 'hoverformat':'.1f', 'tickformat':'.0f',
                              'showline':True, 'linecolor': 'gray',
                              'zeroline':True, 'zerolinewidth':1 , 'zerolinecolor':'whitesmoke',
                              'showgrid': True, 'gridcolor': 'whitesmoke',
                              },
                      yaxis= {'anchor': 'x1', 'title': '', 'hoverformat':'.1f', 'tickformat':'.0f',
                              'showline':True, 'linecolor':'gray',
                              'zeroline':True, 'zerolinewidth':1 , 'zerolinecolor':'whitesmoke',
                              'showgrid': True, 'gridcolor': 'whitesmoke'
                              },
                      updatemenus= [dict(type='buttons',
                                         active=-1, showactive = True,
                                         direction='down',
                                         y=0.5, x=1.1,
                                         pad = {'l':0, 'r':0, 't':0, 'b':0},
                                         buttons=[])],
                      annotations=[],))
        
        # Save template
        pio.templates['multi_strat'] = pio.to_templated(fig).layout.template
        
        return
    
    # %% CLASS PROPERTIES

    # Portfolio or Strategy Returns - should be a pd.DataFrame
    @property
    def portrtns(self): return self.__portrtns
    @portrtns.getter
    def portrtns(self): return self.__portrtns
    @portrtns.setter
    def portrtns(self, x):
        if isinstance(x, pd.Series):
            self.__portrtns = x.to_frame()
        elif isinstance(x, pd.DataFrame):
            self.__portrtns = x
        else:
            raise ValueError('portrtns must be a pandas df or series: {} given'
                             .format(type(x)))

    # Benchmark Returns - should be a pd.DataFrame
    @property
    def bmkrtns(self): return self.__bmkrtns
    @bmkrtns.getter
    def bmkrtns(self): return self.__bmkrtns
    @bmkrtns.setter
    def bmkrtns(self, x):
        if isinstance(x, pd.Series):
            self.__bmkrtns = x.to_frame()
        elif isinstance(x, pd.DataFrame):
            self.__bmkrtns = x
        else:
            raise ValueError('bmkrtns must be a pandas df or series: {} given'
                             .format(type(x)))
            
# %% BIG BANG
            
    def big_bang(self, title=""):
        """ End-to-End Control Function """
        
        # Run Basic Backtest
        self.run_backtest()
        
        # Set up a markdown object and pass self
        # the markdown class contains code to convert backtest metrics to 
        # the plots and HTML needed for an HTML Report
        md = BacktestMarkdwonReport(self)
        md.plot_master()
        markdown = md.markdown_doc(title=title)
        self.markdown_report = markdown
                        
        return markdown

# %% HELPER FUNCTIONS
    
    def _eom(self, x):
        """ Trivial function to ensure End-of-Month Dates in Pandas """
        x.index = x.index + pd.offsets.MonthEnd(0)
        return x
    
# %% BASIC BACKTESTING
        
    def run_backtest(self):
        """ 
        MAIN FUNCTION

        Function will splice port returns with benchmark & Rf returns so we have
        a common time history, then do a series of things:
               - Cumulative Returns
               - Excess Returns (to benchmark)
               - Drawdown and Excess Drawdown
               - Rolling 12m metrics
               - Summary Table - Full Sample Period
               - Summary Table - per_annum except most recent year which is YTD
               - Build "wide" correlation matrix with port rtns, xs returns and
                   all benchmarks specified in self.bmkrtns

        INPUTS not required but the following must have been set:
               - self.portrtns
               - self.bmkrtns
               - self.benchmark
               - self.Rf

        """
        
        # Benchamrk
        # Pull from bmkrtns if provided
        # Pull from bmkrtns if index provided; set as vector of 0 otherwise
        if self.Rb == None:
            bmk = pd.Series(data=0, index=self.portrtns.index, name='BMK')
        else:
            bmk = self.bmkrtns.loc[:,self.Rb]
            bmk.name = 'BMK'
           
        # Risk Free Rate Stuff
        # Pull from bmkrtns if index provided; set as vector of 0 otherwise
        # Be careful about the alignment of dates (indices)    
        if self.Rf == None:
            Rf = pd.Series(data=0, index=self.portrtns.index, name='Rf')
        else:
            Rf = self.bmkrtns.loc[:, self.Rf]
            Rf.name = 'Rf' 
    
        # Consolidated dataframe for risk-free, benchmarks & returns
        # Also set up cumulative returns
        # Rf always at 0, Benchmark always at 1 in dataframe
        self.rtns = pd.concat([Rf, bmk, self.portrtns], axis=1).dropna()
        cr = (1 + self.rtns).cumprod() * 100     # cumulative returns
        self.cum_rtn = cr
       
        # Excess Returns
        # Remember again Rb at 1
        self.xsrtns = self.rtns.subtract(self.rtns.iloc[:, 1], axis='rows')
        self.cum_xs_rtn = cr.subtract(cr.iloc[:,1], axis='rows') + 100

        # drawdown analysis
        self.drawdown = self.rtns2drawdown(alpha=False)
        self.xs_drawdown = self.rtns2drawdown(alpha=True)
        self.drawdown_table = self.drawdown_breakdown(alpha=False)
        self.xs_drawdown_table = self.drawdown_breakdown(alpha=True)

        # Tail risk measures - added by MM
        self.historical_var =self.var_gaussian(level=0.05, cf=False)
        self.modified_var = self.var_gaussian(level=0.05, cf=True)
        self.Historic_CVaR = self.cvar_historic(level=0.05)
        self.omega = self.omega_ratio(threshold=0.0)
        self.downside_risk = self.semivolatility()
      
        # rolling period analysis
        for t in [12]:

            # 12m returns for data & risk free index
            # excess return taken by subtracting the benchmark
            irtn = cr.pct_change(t)
            irtn_xs = irtn.subtract(irtn.iloc[:, 1], axis='rows')
            
            # average rolling  return 
            iMean = self.rtns.rolling(window=t).mean()

            # rolling volatility & downside vol
            iVol = self.rtns.rolling(window=t).std() * np.sqrt(self.freq)
            iDownside = self.rtns[self.rtns < 0].fillna(0).rolling(window=t).std(ddof=0)

            # Ex-Post Tracking Error [std(Rp-Rb)]
            iTE = self.xsrtns.rolling(t).std() * np.sqrt(self.freq)

            # Sharpe Ratio [(Rp-Rb)/vol] & Sortino Ratio
            # Remember Rf at position 0
            iSharpe = irtn.subtract(irtn.iloc[:, 0], axis='rows').divide(iVol, axis='rows')
            iSortino = irtn.subtract(irtn.iloc[:, 0], axis='rows').divide( iDownside, axis='rows')
            
            # Rolling Beta  & hence Treynor ratio          
            iBeta = self.rolling_beta(window=t)
            iTreynor = irtn.subtract(irtn.iloc[:, 0], axis='rows').divide(iBeta, axis='rows')

            # Rolling Value at Risk & CVaR
            alpha = 0.05  # Define the confidence level for rolling CVAR & VaR 
            iVaR = norm.ppf(1-alpha)*(iVol/np.sqrt(self.freq)) - iMean
            iCVaR = alpha**-1*norm.pdf(norm.ppf(alpha))*(iVol/np.sqrt(self.freq)) - iMean

            # save ith data to dictionary
            self.rolling[t] = dict(vol=iVol,
                                   downside_vol = iDownside,
                                   rtn=irtn,
                                   xsrtn=irtn_xs,
                                   te=iTE,
                                   sharpe=iSharpe,
                                   sortino=iSortino,
                                   beta =iBeta,
                                   treynor = iTreynor,
                                   VaR = iVaR,
                                   CVaR = iCVaR
                                   )

        ## Run summary table & annualised summary and ingest
        # summary is a dataframe with lots of metrics over the whole period
        # annualised is a dict of metrics, where values are dfs with ann summary
        # dataframe has the years as index and strategies in columns
        self.summary = self.backtest_summary()
        self.summary_pa = self.per_annum()

        # Extended Correlation Matrix
        # Use BMK, PORT, PORT_XS_RTNS & the bmkrtns indices to form corr matrix
        # Some minor adjustments to remove Rf from 1st column
        rtns_wide = pd.concat([self.rtns.iloc[:,1:], self.xsrtns.iloc[:, 2:]], axis=1)
        rtns_wide.columns = list(self.xsrtns.columns)[1:] + list(self.xsrtns.columns + '_XS')[2:]
        rtns_wide = pd.concat([rtns_wide, self.bmkrtns], axis=1).dropna()
        self.rtns_wide = rtns_wide
        self.corr = rtns_wide.corr()

        return
    
    def rtns2drawdown(self, alpha=True):
        """ Returns-to-Drawdown Timeseries 
        
        NB/ Rebased to 0 not 100
        """
    
        # Need to select a method for drawdown
        # if alpha is True use excess returns, otherwise returns        
        # Remove risk free column  
        rtns = self.xsrtns if alpha else self.rtns
        rtns = rtns.iloc[:,1:]
        
        
        dd = 1 + rtns         # add 1 to monthly rtns
        dd.iloc[0,:] = 100    # rebase to 100
        
        # iterate through each time period
        # create an index series with a max of 100
        for i, d in enumerate(dd.index):
            
            # ignore 0th because we need the i-1 time period
            if i == 0:
                continue
            
            ix = dd.iloc[i-1] * dd.iloc[i,:]    # index level for i
            ix[ix > 100] = 100                  # reset to 100 if > that
            dd.iloc[i,:] = ix                   # populate drawdown dataframe
            
        return (dd - 100) / 100    # set to zero & percentages
    
    def drawdown_breakdown(self, alpha=True, dd_threshold=0):
        """ Drawdowns Details by Individual Drawdown
        
        Builds table which breaks out each individual drawdown period from a
        timeseries of drawdowns (set with base at zero NOT 100). Table data 
        currently shows the date drawdown starts, throughs & ends as well as 
        the number of months in total, top to bottom & recovery as well as the
        max drawdown itself.
        
        INPUTS:
            alpha: True(default)|False the function takes the drawdown ts from 
                self.xs_drawdown or self.drawdown depending on if alpha.
                NB/ MUST RUN self.rtns2drawdown FIRST OR THIS WILL FAIL
            dd_threshold: +ve number; excludes drawdowns less than this level.
        
        """
        
        # determine if we need table of excess drawdowns or just drawdowns
        # the table determines if we need to start on the 0th or 1st column
        # (no point doing excess returns on a benchmark)
        if alpha:
            dd = self.xs_drawdown
            n_start = 1
        else:
            dd = self.drawdown
            n_start = 0
        
        # dummy output dataframe
        df = pd.DataFrame()
    
        # iterate through strategies in port
        # start at 
        for p in dd.columns[n_start:]:
    
            ix = dd[p]
    
            # find index of series where value == 0
            # can use this to to create sub-series of drawdowns
            idx = np.argwhere(ix.values == 0)
            idx = list(idx.flatten())    # flatten to list (makes life much easier)
            
            # because we are searching for 0s we won't get an index if the end
            # of the current timeseries is still in drawdown
            # fudge by checking if the last index == index of final obs
            # if not add the index of final obs to the list
            if idx[-1] != int(len(ix)):
                idx.append(len(ix))
    
            for i, v in enumerate(idx):
    
                # relative index means we start by looking back
                # thus ignore the first zero we find
                if i == 0:
                    continue
    
                z = ix.iloc[(idx[i-1]+1):(idx[i]+1)]
    
                # ignore blanks (which will be periods of positive performance)
                if len(z) > 1:
    
                    # create dictionary with info from drawdown period
                    start=z.index[0]
                    end=z.index[-1]
                    trough=z.idxmin()
                    
                    # subset the series to just the ith drawdown
                    idd = dict(start=start, end=end, trough=trough,
                               length=z.count(),
                               fall=ix.loc[start:trough].count(),
                               recovery=ix.loc[trough:end].count(),
                               drawdown=z.min(),)
    
                    # This is a wrinkly
                    # We forced an index if the series is still in drawdown
                    # need to set the exit & the recovery to nan
                    if v == idx[-1] and ix[-1] != 0:
                        idd['recovery'] = np.nan
                        idd['end'] = np.nan
    
                    # Thresholds
                    if abs(idd['drawdown']) < dd_threshold:
                        continue
    
                    # add to output dataframe
                    df = pd.concat([df, pd.Series(idd, name=p)], axis=1)
                    
        return df.T
    
    
    def var_historic(self, level=0.05):
        '''
        Returns the (5-level)% VaR using historical method. 
        By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.

        '''  
        mu= self.rtns.mean()
        sigma = self.rtns.std(ddof=0)
        VaR = norm.ppf(1-level)*sigma - mu
        return VaR
            
    def var_gaussian(self, level=0.05, cf=False):
        
        '''
        Returns the (1-level)% VaR using the parametric Gaussian method. 
        By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
        The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
        modified VaR using the Cornish-Fisher expansion of quantiles.
        '''   
        # alpha-quantile of Gaussian distribution 
        za = scipy.stats.norm.ppf(level,0,1) 
        
        if cf:
            S = skew(self.rtns)
            K = kurtosis(self.rtns)
            za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/36    
        return -( self.rtns.mean() + za * self.rtns.std(ddof=0) ) 
    
    def cvar_historic(self, level=0.05):
        '''
        Computes the (1-level)% Conditional VaR (based on historical method).
        By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.

        '''
        mu= self.rtns.mean()
        sigma = self.rtns.std(ddof=0)
        CVaR = level**-1*norm.pdf(norm.ppf(level))*sigma - mu
        return CVaR 
            
    def semivolatility(self):
        '''
        Returns the semivolatility of a series, i.e., the volatility of
        negative returns
        '''
        return self.rtns[self.rtns<0].std(ddof=0) 
    
    
    def omega_ratio(self, threshold=0.0):
        """
        calculate the omega ratio
        :param returns: the returns
        :param threshold:  the threshold
        """
        # Calculate the difference between the threshold and the returns
        diff = self.rtns-threshold
        # Get sum of all values excess return above 0
        PositiveSum = diff[diff >0].sum()
        # Get sum of all values excess return below 0
        NegativeSum = diff[diff <0].sum()
        
        omega = PositiveSum/(-NegativeSum)
        # Return the sum of the different to the power of order
        return omega
    
    def rolling_beta(self, window=12):
        """
        calculate the rolling betas against specific benchmark
        :param window: the rolling window 
        """
        # Define the dependent variable : In our case it will be the Benchmark
        endog_var = self.rtns.iloc[:, 0:]
        
        # Define the independent  variable  
        exog_var = sm.add_constant(self.rtns.iloc[:,1]) # Calculate the beta against a given  market/benchmark
        
        # loop through the colums of the indepedent variables
        beta_dict ={}    # dummy dict for rolling betas
        for col in endog_var.columns:
            
            # This function runs rolling ols for defined rolling window
            dx = pd.DataFrame(RollingOLS(endog_var[col],
                                         exog_var,
                                         window=window).fit().params) 
            
            dx['key'] = str(col)    # add key name column
            beta_dict[col] = dx     # append to dictionary
        
        # # adding the key as a column to each DataFrame
        # for key, df in rol_beta.items():
        #     # create a column called "key name"
        #     df['key_name'] = key
        
        # make a list just using the values in the dictionary
        #lst = list(rol_beta.values())
        #df = pd.concat(lst) # create one new dataframe of the all the dataframe inside dictionary 
        #df_new =df.drop(df.columns[0], axis=1) # remove constant column from the rolling regression analysis. It will always be the first column
        #df_new=df_new.reset_index() # Reset the index 
        #key =df_new['key_name'].tolist() # store Column names in a list . This is done so that pivot keeps the orginal order of columns 
        #key = list(dict.fromkeys(key))
        
        # unstack the dataframe 
        #roll_beta = df_new.pivot(index='index', columns='key_name', values='BMK').reindex(columns=key)
        
        # New DataFrame which stacks all the dataframes inside the dictionary
        # we can drop the constant column for roll regresison; always column 0
        df = pd.concat(list(beta_dict.values())).drop(columns={'const'}).reset_index()
        
        # pivot output so index == Dates & columns are [Rf, BMK, Strategies+]
        # reindex so output order is the same as input dataframe column order
        roll_beta = (df.pivot(index='Dates', columns='key', values='BMK').
                     reindex(columns=list(self.rtns.columns)))

        return roll_beta

    def backtest_summary(self):
        """ Summary Table for the Whole Sample Period 
        
        """
        
        df = pd.DataFrame()
        
        # Annualised Total Return, Vol & Risk-adjusted-return
        df['TR'] = (self.cum_rtn.iloc[-1,:]/100)**(self.freq/len(self.cum_rtn)) - 1
        df['Vol'] = self.rtns.std() * np.sqrt(self.freq)
        df['Downside_Vol']=self.downside_risk*np.sqrt(self.freq)
            
        # Beta, Ex-Post Tracking Error & Information Ratio
        df['Beta'] = self.rtns.cov().iloc[:,1] / self.rtns.iloc[:,1].var()
        df['TE'] = self.xsrtns.std() * np.sqrt(self.freq)
        df['IR'] = (df.TR - df.TR[1]) / df.TE
        
        # Risk adjusted return metrics; Sharpe, RaR and Sortino
        df['Sharpe'] = (df.TR - df.TR[0]) / df.Vol
        df['RaR'] = df.TR / df.Vol
        df['Sortino'] = ((self.cum_xs_rtn.iloc[-1,:]/100)**(self.freq/len(self.cum_xs_rtn)) - 1)/df.Downside_Vol
        
        # Calculates Average Return 
        df['Avg Return'] = self.rtns[self.rtns != 0].dropna().mean()
        
        # calculates the average win/loss return/trade return for a period
        df['avg_win'] = self.rtns[self.rtns > 0].dropna().mean()
        df['avg_loss'] = self.rtns[self.rtns < 0].mean()
        
        # measures the payoff ratio (average win/average loss)
        df['Payoff'] = df.avg_win/abs(df.avg_loss)
        
        # Drawdown Metrics over full period
        # drawdown, xs-drawdowns, average drawdown period & xs-period (in days)
        df['avg_drawdown'] = self.drawdown.mean()
        df['avg_XS_drawdown'] = self.xs_drawdown.mean()
        df['avg_drawdown_days']=self.drawdown_breakdown(alpha=False)['length'].mean() 
        df['avg_XS_drawdown_days']=self.drawdown_breakdown(alpha=True)['length'].mean() 
            
        # Measure the skewness & kurtosis of the returns
        df['Skew'] = self.rtns.skew()
        df['Kurtosis'] = self.rtns.kurt()
            
        #Tail risk Measurements
        df['Historic_VaR'] = -1*self.historical_var
        df['Historic_CVaR']= -1*self.Historic_CVaR
        df['Modified_VaR'] = -1*self.modified_var
        
        # Treynor ratio determines how much excess return was generated for each unit of systematic risk taken on by a portfolio
        df['Treynor_Ratio'] = (df.TR - df.TR[0]) / df.Beta
        
        #  measures the ratio between the right (95%) and left tail (5%)
        df['tail_ratio']=abs(self.rtns.quantile(0.95) / self.rtns.quantile(0.05))
        
        # Drawdown Analysis
        df['Max_Drawdown'] = self.drawdown.min(axis=0)
        df['Max_XS_DD'] = self.xs_drawdown.min(axis=0)
        df['Hitrate'] = self.xsrtns[self.xsrtns > 0].count() / self.rtns.count()
        df['xs_mean'] = self.xsrtns.mean()
        df['xs_worst'] = self.xsrtns.min()
        df['xs_best'] = self.xsrtns.max()
        
        # the likelihood of losing all one's investment capital
        df['risk_of_ruin'] = ((1 - df.Hitrate) / (1 + df.Hitrate)) ** len(self.rtns)
        
        #  measures the profit ratio (win ratio / loss ratio)
        df['profit_factor'] = df.Hitrate/(1 - df.Hitrate)
        
        # measures how fast the strategy recovers from drawdowns
        df['recovery_factor'] = (self.cum_rtn.iloc[-1,:]/100)/abs(df.Max_Drawdown)
        
        # calculate the upside potential ratio
        #df['upside_potential_ratio'] = self.hpm/(math.sqrt(self.lpm2))
        
        # Omega Ratio
        df['Omega_ratio'] =  self.omega
        df['Modified_Sharpe_Ratio']= (df.TR - df.TR[0])/abs(df.Modified_VaR)
        
        # Calmar ratio
        df['Calmar Ratio'] = df.TR / abs(df.Max_Drawdown)
        
        # Remove Risk Free Rate from summary table
        self.Rf_obs_rtn = df.loc['Rf', 'TR']
        self.summary = df.T.iloc[:, 1:]
        return self.summary
    
    
    def per_annum(self):
        """ Convert Return Stream to Per Annum Metrics
        
        NB/ for current year we calculate YTD rather than annualised or 12m
        """
         
        # Requires only the returns dataframe
        # Rf in [0] and Rb in [1]
        x = self.rtns
        y = self.rtns[self.rtns < 0].fillna(0) # only requires return dataframe that holds negative returns
        
        pa = dict.fromkeys(['rtn', 'alpha', 'xscash', 'vol','downside_vol', 'te', 'sharpe', 'ir','sortino','Beta','treynor'])
        
        # create group object which has years as keys
        # find index of last month; can't just do annual or we miss YTD
        grp = x.index.groupby(x.index.year)
        idx = [v[-1] for v in grp.values()]    # index of last month
        yrs = grp.keys()                       # list of years
        
        # Return - Annual & current YTD
        # 1st ret2px, then subset dates and calc return
        rtn = (1 + x).cumprod().loc[idx, :].pct_change()
        rtn.index = yrs    # relabel indices to years (from timestamp)
        pa['rtn'] = rtn
        
        # Volatility & downside vol - fairly simple
        pa['vol'] = x.groupby(x.index.year).std() * np.sqrt(12)
        pa['downside_vol'] = y.groupby(y.index.year).std() * np.sqrt(12)
        
        # Beta : Used the Rolling beta to extract the Betas at the end of each year
        beta=self.rolling_beta(window=12)
        pa['Beta'] = beta[beta.index.isin(idx)]
        pa['Beta'].index = pd.DatetimeIndex(pa['Beta'].index).year
        
        # Alpha & Excess-Cash Return
        # Remember Rf in posn 0 & Rb in posn 1
        pa['xscash'] = rtn.subtract(rtn.iloc[:,0], axis='rows')
        pa['alpha'] = rtn.subtract(rtn.iloc[:,1], axis='rows')
        
        # Tracking Error
        # Can't use rtn above because that is annualised
        # Need to create monthly series of alpha stream
        xsrtn = x.subtract(x.iloc[:,1], axis='rows')
        pa['te'] = xsrtn.groupby(x.index.year).std() * np.sqrt(12)
        
        # Sharpe & IR therefore easy to calculate
        pa['sharpe'] = pa['xscash'] / pa['vol']
        pa['ir'] = pa['alpha'] / pa['te']
        pa['sortino'] = pa['xscash'] / pa['downside_vol']
        pa['treynor'] = pa['xscash'] / pa['Beta']
        
        self.summary_pa = pa
        return pa

# %% PLOTLY PLOTS
    
    def _px_addsource(self, fig, x=1, y=-0.125, align='right'):
        return fig.add_annotation(
                text="Source: STANLIB Multi-Strategy".format(),
                xref='paper', yref='paper',
                x=x, y=y, ax=0, ay=0,
                align=align)
    
    def plot_index(self, df, title="", benchmark=True, risk_free=False,
                   yfmt=['.0f', '.2f'], ytitle='Port', height=0,
                   source=False, y_src=-0.15):
        """ Basic Line Plot in Backtester"""
        
        # Remember the 1st column is a Risk-Free rate
        Rf = df.iloc[:,0]     # Risk Free
        df = df.iloc[:,1:]    # Benchmark & Simulations

        # Plot basic line
        fig = px.line(df, title=title, labels={'variable':'Port:'}, template='multi_strat', )
        
        # Append Risk-Free Line if Required
        if risk_free:
            fig.add_scatter(x=Rf.index, y=Rf, name="Rf",
                            line={'color':'black', 'dash':'dot','width': 0.75})
        
#        # Hide benchmark if required
         # It seems like the error is ocuuring at this line. It must have something do with benchmark having zero data or zero output
        if not benchmark:
            fig.data[0]['visible'] = 'legendonly'    # hide bmk
        
        fig.update_layout(
                yaxis= {'anchor':'x1','title':ytitle, 'tickformat':yfmt[0], 'hoverformat':yfmt[1], },
                xaxis= {'anchor':'y1','title':'', 'hoverformat':'%b-%y', 'tickformat':'%b-%y',},)
        
        if height != 0:
            fig.update_layout(height=height)
        
        if source:
            fig = self._px_addsource(fig, y=y_src)
        
        return fig
    
    
    def plot_ridgeline(self, df, title='Ridgeline KDE Distributions',
                       side='positive', meanline=True, box=False, width=3,
                       template='multi_strat', 
                       source=False, y_src=-0.15,
                       **kwargs):
        """ Simplified KDE from bootstrapper """
        
        # Remember the 1st column is a Risk-Free rate
        #Rf = df.iloc[:,0]     # Risk Free
        df = df.iloc[:,1:]    # Benchmark & Simulations
        
        n = len(df.columns)
        
        # create a blended colours list- here is teal to purple
        if n > 1:
            from plotly.colors import n_colors
            colors = n_colors('rgb(0, 128, 128)', 'rgb(128, 0, 128)', n, colortype='rgb')
        else:
            colors = ['rgb(0, 128, 128)']
        
        # blank plotly express template
        fig = px.scatter(title=title, template=template)    
        for i, v in enumerate(df):             # add violin plots as traces
            fig.add_trace(go.Violin(x=df.iloc[:,i],
                                    line_color=colors[i],
                                    line_width=1,
                                    name=v,
                                    spanmode='soft',))
        
        # convert from violins to horizontal kde charts 
        fig.update_traces(orientation='h', 
                          side=side,
                          meanline_visible=meanline,
                          width=width,
                          box_visible=box)
        
        # update layouts
        fig.update_layout(
            yaxis= {'anchor':'x1', 'title':'Simulation', 'hoverformat':'.1%', 'tickformat':'.0%',},
            xaxis= {'anchor':'y1', 'title':'Annualised Return', 'hoverformat':'.1%', 'tickformat':'.0%',})
        
        if source:
            fig = self._px_addsource(fig, y=y_src)
        
        return fig

    def plot_histo(self, df, title='', opacity=0.5, benchmark=False,
                   source=False, y_src=-0.15):
        """ Basic Histogram """
        
        # Remember the 1st column is a Risk-Free rate
        #Rf = df.iloc[:,0]     # Risk Free
        df = df.iloc[:,1:]    # Benchmark & Simulations
    
        fig = px.histogram(df, title=title, histnorm='probability', 
                           opacity=opacity, template='multi_strat')
        
        if benchmark != True:
            fig.data[0]['visible'] = 'legendonly'    # hide bmk from histogram
        
        fig.update_layout(barmode='overlay')
        fig.update_layout(
            yaxis= {'anchor':'x1','title':'Probability', 'tickformat':'.0%', 'hoverformat':'.2%', },
            xaxis= {'anchor':'y1','title':'Excess Return', 'tickformat':'.1%', 'hoverformat':'.2%', },)
        
        if source:
            fig = self._px_addsource(fig, y=y_src)
        
        return fig

    def plot_regression(self, title='', alpha=True,
                        source=False, y_src=-0.15):
        """ CAPM Style Regression Plot
        
        Plots the benchmark on the x-axis & port(s) on the y-axis; 
        OLS regression line plotted through

        IMPORTANT:
            function takes input dataframe from self. Therefore in order to 
            run you need to have already run a backtest function which stores
            self.xsrtns or self.rtns

        INPUTS:
            alpha: True(default)|False decides between xsrtns or rtns dfs 
        
        """
        
        # stack either the returns or excess returns
        # rename columns as required
        # Also remember to remove risk free column
        if alpha:
            y = self.xsrtns.iloc[:,1:].stack().reset_index()
            ytitle='Alpha'
            benchmark=False
        else:
            y = self.rtns.iloc[:,1:].stack().reset_index()
            ytitle='Port Return'
            benchmark=False
        
        y.columns = ['Dates', 'Port', 'Returns']    # rename columns
    
        # Repmat benchmark returns & Match columns
        # This is so we can stack - so we can then concat
        x = pd.concat([self.rtns['BMK']] * (len(self.xsrtns.columns)-1), axis=1)
        x.columns = self.xsrtns.columns[1:]   # [1:] excludes Rf
        x = x.stack().reset_index()
        x.columns = ['Dates', 'Port', 'Mkt']
    
        # Merge things together so we have an x, y & colour column
        z = pd.concat([x,y['Returns']], axis=1)
        
        # plot scatter with OLS
        fig = px.scatter(z, title=title, 
                          x='Mkt', y='Returns', color='Port',
                          trendline="ols",
                          template='multi_strat')
            
        fig.update_layout(
                    yaxis= {'anchor':'x1','title':ytitle, 'tickformat':'.1%', 'hoverformat':'.2%', },
                    xaxis= {'anchor':'y1','title':'Benchmark Return', 'tickformat':'.1%', 'hoverformat':'.2%', },)
        
#        if not benchmark:
#            fig.data[0]['visible'] = 'legendonly'    # hide bmk
            
        if source:
            fig = self._px_addsource(fig, y=y_src)
        
        return fig

    
    def plot_hitrate(self, df, title='', binary=True,
                     source=False, y_src=-0.15):
        """ Hitrate Heatmap
        
        Plots Months x Years Heatmap, either as returns or binary outcome
        
        INPUT:
            df: pd.DataFrame with each columns a series of returns
            binary: True(default)|False map the returns or switch to 1/0
                depending on if the monthly return was positive or negative
        """
        
        # Use crosstab to break pd.Series to pd.DataFrame with months x years
        # Cols will be done alphabetically so we manually reorder dataframe
        plots = pd.crosstab(df.index.year,
                            df.index.strftime("%b"),
                            df.values,
                            aggfunc='sum',
                            rownames=['years'],
                            colnames=['months'])
        
        # Re0order because crosstab will spit out in alphabetical order
        plots = plots.loc[:,['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        
        # Convert excess returns to hit/miss
        if binary:
            plots = plots.applymap(lambda x: 1 if x >= 0 else x)
            plots = plots.applymap(lambda x: 0 if x <= 0 else x)
        
        # Plot
        fig = px.imshow(plots, x=plots.columns.to_list(), y=plots.index.to_list(),
                        title=title,
                        labels=dict(x='Month', y='Year', color='Hit-or-Miss'),
                        color_continuous_midpoint=0,
                        aspect='auto', template='multi_strat')
        
        # Colourscale stuff
        fig.update_traces(dict(colorscale='Tealrose', reversescale=True, showscale=False, coloraxis=None),)
        
        if source:
            fig = self._px_addsource(fig, y=y_src)
        
        return fig

    # def plot_hitrates(self,
    #                   min_count=3,
    #                   show=False,
    #                   plotly2html=True, plotlyjs='cdn',
    #                   plot_height=450, plot_width=850):
        
    #     """ Combined Function Charts & Annual Table
        
    #     Charts are the Year x Month, binary, hit-or-miss heatmap and
    #     Table is the annualised hit rate per year - in a styled dataframe
        
    #     IMPORTANT:
    #         requires bactest functions have to be run because needs
    #         self.xsrtns & self.rtns dataframes to exist
    #         uses self.plot_hitrates() function
        
    #     INPUTS:
    #         min_count: 3(default) is min. months req to get a per-year number
            
    #     OUTPUT:
    #         dict() with key 'annual' for styled df & others will be port name
    #         i.e. {'annual':styled_df, 'PORT1':plotly_fig}
        
    #     """
        
    #     plots = dict()        # dummy dictionary for plotly & dataframe
    #     df = pd.DataFrame()   # dummy dataframe for table
        
    #     # iterate through each portfolios alpha
    #     # could be done as a matrix but complexity isn't worth the speed
    #     # remember to remove risk-free column zero
    #     for i, p in enumerate(self.xsrtns.iloc[:,1:]):
            
    #         if i == 0:
    #             continue
            
    #         ## PLOTLY
    #         # Get the Year x Month Hitrate Plot
    #         plots[p] = self.plot_hitrate(self.xsrtns[p],
    #                                       title="Hit Rate Heatmap: {}".format(p),
    #                                       binary=True,
    #                                       source=False, y_src=-0.15)
            
    #         ## TABLE
    #         # Calc the average annual across backtest things
    #         # use crosstab again to break down by year and month
    #         # then we can sum across the months
    #         ix = self.xsrtns.loc[:, p]
    #         ix = pd.crosstab(index=ix.index.year,
    #                           columns=ix.index.strftime("%b"),
    #                           values=ix.values,
    #                           aggfunc='sum',
    #                           rownames=['years'],
    #                           colnames=['months'])
    
    #         # Map excess returns to hit or miss
    #         ix = ix.applymap(lambda x: 1 if x >= 0 else x)
    #         ix = ix.applymap(lambda x: -1 if x <= 0 else x)
    
    #         # Hit rate by month & rename result series
    #         ihr = ix[ix==1].sum(axis=1) / ix.count(axis=1)
            
    #         # May not want to distort data if we don't have a minimum no obs
    #         # by default we use 3-months
    #         if min_count > 0:
    #             ihr[~(ix.count(axis=1) >= min_count)] = np.nan
    #         ihr.name = p
    
    #         df = pd.concat([df, ihr.to_frame()], axis=1)
        
    #     if show:
    #         for k in plots:
    #             plots[k].show()
        
    #     if plotly2html:
    #         for k, v in plots.items():
    #             plots[k] = v.to_html(full_html=False,
    #                                   include_plotlyjs=plotlyjs,
    #                                   default_height=plot_height,
    #                                   default_width=plot_width,
    #                                   )
                  
    #     plots['annual'] = self.pretty_panda(df.reset_index())\
    #             .format(formatter="{:.0f}", subset=pd.IndexSlice[:, df.columns[0]])\
    #             .format(formatter="{:.1%}", subset=pd.IndexSlice[:, df.columns[0:]])\
    #             .background_gradient('RdYlGn', vmin=0.2, vmax=0.8, subset=pd.IndexSlice[:, df.columns[0:]])\
    #             .highlight_null(null_color='white')\
        
    #     return plots
    
    def plot_correl(self, cor=None, title='Correlation Matrix', aspect='auto',
                    colorscale='Tealrose', reversescale=False, **kwargs):
        """ Plotly Heatmap with Overlay annotations
        
        NB/ DIRECT RIP FROM BOOTSTRAP - need to consolidate these in Overplot
        """
            
        # Pull correlation matrix from Bootsrap class
        if cor is None:
            cor = self.corr
            
        ## Basic plotly express imshow heatmap
        fig = px.imshow(cor,
                        x=cor.index, y=cor.index,
                        labels={'color':'Correlation'}, 
                        title=title,
                        color_continuous_midpoint=0,   # change for VCV
                        aspect=aspect,    
                        **kwargs)
                
            ## Formatting
        fig.update_layout(margin = {'l':25, 'r':50, 'b':0, 't':50},)
            
            # format to 2dp by editing the z axis data
        fig['data'][0]['z'] = np.round(fig['data'][0]['z'], 2)
            
            # Heatmap colour - I rather like Tealrose
        fig.update_traces(dict(colorscale=colorscale, reversescale=reversescale,
                               showscale=False, coloraxis=None),)
            
            # By default plotly imshow doesn't give values, so we append
            # Each value is a unique annotation
            # iterate through columns & rows (which is a bit silly)
        N = cor.shape[0]
        for i in range(N):
            for j in range(N):
                fig.add_annotation(text="{:.2f}".format(cor.iloc[i,j]),
                                       font={'color':'black', 'size':9},
                                       xref='x',yref='y',x=i,y=j,ax=0,ay=0)
            
        return fig
    
    
    def plot_correl_animation(self, title="Rolling Correlation", subset=True, years=1):
        """ Animated Correlation Matrix
        
        Creates plotly animation with rolling period correlation matrix
        
        INPUTS:
            years: 1 as default; gets multiplied by self.freq for data range
            
        DEVELOPMENT:
            Having problems with the axis labels overlapping the title

        """
        
        wide = self.rtns_wide      # pull wide reutrns & xs return series
        if subset:
            wide = wide.iloc[:,:(len(self.rtns.columns[1:])*2-1)] 
        corr = wide.corr()         # basic correlation matrix
        
        n = self.freq * years        # data rolling window
        
        # Initial Correlation plot
        fig = px.imshow(corr, x=corr.columns, y=corr.index,
                        title=title, aspect='auto',
                        color_continuous_midpoint=0, zmin=-1, zmax=1)
        
        # Update Main Plot
        fig.update_traces(dict(colorscale='Tealrose', reversescale=False,
                               showscale=True, coloraxis=None),)
        fig.update_layout(margin = {'l':25, 'r':50, 'b':10, 't':75}, font_size=10)
        fig.update_xaxes(side="top")    # sort of struggling with overalapping
        
        # Add play & pause buttons
        fig["layout"]["updatemenus"] = [
                {"buttons":[{"args":[None,{"frame": {"duration": 500, "redraw": True},
                                           "fromcurrent": True,
                                           "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                                           "label": "Play",
                                           "method": "animate"},
                            {"args":[[None],{"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}],
                                             "label": "Stop",
                                             "method": "animate"}],
                 "direction": "left", "pad": {"r": 0, "t": 20},
                 "showactive": False, "type": "buttons",
                 "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"}]
        
        ### Animation Stuff
        frames = []
        sliders = {'yanchor': 'top',
                   'xanchor': 'left', 
                   'currentvalue': {'prefix':'{}m Correlation: '.format(n),
                                    'font':{'size': 10}, 'xanchor': 'left',},
                   'transition': {'duration': 10, 'easing': 'linear'},
                   'pad': {'b': 0, 't': 0}, 'len': 0.88, 'x': 0.12, 'y': 0,
                   'steps':[]}
        
        for i, v in enumerate(wide.index[n:], n):
    
            c = wide.iloc[i-n:i,:]         # subset ith data
            c = c.corr()                    # subset correlation matrix
            label = '{:%m/%y}'.format(v)    # string label - used to link slider and frame (data things)
            
            # Add ith correlation matrix to frames list
            frames.append({'name':i, 'layout':{},
                           'data': [dict(type='heatmap',
                                         x=c.index, y=c.index,
                                         z=c.values.tolist(),
                                         zmin=-1, zmax=1)]})
            
            # Add ith thing to sliders
            sliders['steps'].append({'label':label, 'method': 'animate', 
                                     'args':[[i], {'frame':{'duration': 0, 'easing':'linear', 'redraw': True},
                                                   'transition':{'duration': 0, 'easing': 'linear'}}],})
        
        # Append Frames & Sliders to 
        fig['frames'] = frames
        fig['layout']['sliders'] = [sliders]
        
        return fig

    
#     def plot_master(self, plotly2html=True, plotlyjs='cdn',
#                     plot_height=450, plot_width=850):
#         """ Aggregation Function that runs ALL plots
        
#         These are saved in a big dictionary self.plots
        
#         INPUTS:
#             all related to if we want to save as HTML for output
#             this is required for markdown but less useful if used in an app
#         """
        
#         plots = dict()   # dummy dictionary to hold plots
        
#         # Total Return & Excess Return
#         plots['tr'] = self.plot_index(self.cum_rtn,
#                                       title='Cumulative Returns',
#                                       ytitle='Index Level', 
#                                       risk_free=True,
#                                       source=True, y_src=-0.125)
        
#         plots['xsrtn'] = self.plot_index(self.cum_xs_rtn,
#                                       title='Excess Returns',
#                                       ytitle='Excess Returns',
#                                       benchmark=False,
#                                       source=True, y_src=-0.125)
        
#         # Return Distributions
#         plots['kde_rtns'] = self.plot_ridgeline(
#                                     self.rtns,
#                                     title='Ridgeline KDE Distributions: Returns',
#                                     source=True, y_src=-0.125)
        
#         plots['kde_alpha'] = self.plot_ridgeline(
#                                     self.xsrtns.iloc[:, 1:],
#                                     title='Ridgeline KDE Distributions: Excess Returns',
#                                     source=True, y_src=-0.125)
        
        
#         # Regression Charts
#         plots['regression_rtn'] = self.plot_regression(
#                                             alpha=False,
#                                             title='Return Regression: Port Returns',
#                                             source=True, y_src=-0.125)
# #        
#         plots['regression_alpha'] = self.plot_regression(
#                                             alpha=True,
#                                             title='Return Regression: Excess Returns',
#                                             source=True, y_src=-0.125)
        
#         plots['histogram'] = self.plot_histo(self.xsrtns,
#                                     title='Excess Return Distribution',
#                                     source=True, y_src=-0.125)
        
        
#         # Drawdown Charts
#         plots['drawdown'] = self.plot_index(self.drawdown,
#                            title='Drawdown of Returns',
#                            yfmt=['.1%', '.2%'], ytitle='Drawdown',
#                            benchmark=True,
#                            source=True, y_src=-0.125)
        
#         plots['xs_drawdown'] = self.plot_index(self.xs_drawdown,
#                            title='Drawdown of Excess Returns',
#                            yfmt=['.1%', '.2%'], ytitle='Drawdown',
#                            benchmark=False,
#                            source=True, y_src=-0.125)
        
#         # Rolling Plots
#         # Rolling Period Charts
#         plots['roll_rtn'] = self.plot_index(self.rolling[12]['rtn'],
#                                             title='Rolling Return: 12m',
#                                             yfmt=['.0%', '.2%'],
#                                             ytitle='Return',
#                                             height=350,
#                                             source=True, y_src=-0.15)
        
#         plots['roll_xsrtn'] = self.plot_index(self.rolling[12]['xsrtn'],
#                                             title='Rolling Excess Return: 12m',
#                                             yfmt=['.0%', '.2%'],
#                                             ytitle='Alpha',
#                                             benchmark=False, height=350,
#                                             source=True, y_src=-0.15)
        
#         plots['roll_vol'] = self.plot_index(self.rolling[12]['vol'],
#                                             title='Rolling Volatility: 12m',
#                                             yfmt=['.0%', '.2%'],
#                                             ytitle='Volatility',
#                                             height=350,
#                                             source=True, y_src=-0.15)
        
#         plots['roll_downside_vol'] = self.plot_index(self.rolling[12]['downside_vol'],
#                                             title='Rolling Downside Volatility: 12m',
#                                             yfmt=['.0%', '.2%'],
#                                             ytitle='Downside Volatility',
#                                             height=350,
#                                             source=True, y_src=-0.15)        
        
#         plots['roll_te'] = self.plot_index(
#                                  self.rolling[12]['te'],
#                                  title='Rolling ex-Post TE: 12m',
#                                  yfmt=['.1%', '.2%'], ytitle='Tracking Error',
#                                  benchmark=False, height=350,
#                                  source=True, y_src=-0.15)

#         plots['roll_sharpe'] = self.plot_index(
#                                   self.rolling[12]['sharpe'],
#                                   title='Sharpe Ratio: 12m',
#                                   yfmt=['.1f', '.2f'], ytitle='Sharpe Ratio',
#                                   benchmark=False, height=350,
#                                   source=True, y_src=-0.15)
        
#         plots['sortino'] = self.plot_index(
#                                 self.rolling[12]['sortino'],              
#                                 title='Rolling Sortino: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='Sortino',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)
        
#         plots['roll_rar'] = self.plot_index(
#                                  self.rolling[12]['xsrtn'] / self.rolling[12]['vol'],
#                                  title='Risk Adjusted Return: 12m',
#                                  yfmt=['.1f', '.2f'], ytitle='Information Ratio',
#                                  benchmark=False, height=350,
#                                  source=True, y_src=-0.15)

#         plots['roll_ir'] = self.plot_index(
#                                 self.rolling[12]['xsrtn'] / self.rolling[12]['te'],              
#                                 title='Rolling Information Ratio: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='IR',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)
        
#         plots['beta'] = self.plot_index(
#                                 self.rolling[12]['beta'],              
#                                 title='Rolling Beta: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='Beta',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)
        
#         plots['treynor'] = self.plot_index(
#                                 self.rolling[12]['treynor'],              
#                                 title='Rolling Treynor Ratio: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='Treynor',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)
        
        
#         plots['VaR'] = self.plot_index(
#                                 self.rolling[12]['VaR'],              
#                                 title='Rolling Value-at-Risk: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='VaR',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)   
            
#         plots['CVaR'] = self.plot_index(
#                                 self.rolling[12]['CVaR'],              
#                                 title='Rolling Expected Shortfall: 12m',
#                                 yfmt=['.1f', '.2f'], ytitle='CVaR',
#                                 benchmark=False, height=350,
#                                 source=True, y_src=-0.15)   
        

#         # Correlation
#         plots['correl_wide'] = self.plot_correl(self.corr)
#         plots['correl_animation'] = self.plot_correl_animation(subset=True, years=1)
        
#         # Convert to HTML
#         if plotly2html:
#             for k, v in plots.items():
#                 plots[k] = v.to_html(full_html=False,
#                                      include_plotlyjs=plotlyjs,
#                                      default_height=plot_height,
#                                      default_width=plot_width,
#                                      )   
#         # Hitrate Plots
#         # These come after we've already converted most to plotly
#         # that is because this is an embedded dictionary & we've already
#         # converted the plotly
#         plots['hitrate'] = self.plot_hitrates(show=False,
#                                               plotly2html=plotly2html,
#                                               plotlyjs=plotlyjs,
#                                               plot_height=300,
#                                               plot_width=750)
# #        
#         self.plots = plots
        
#         return plots

# %% REPORTING

#     def pretty_panda(self, df):
#         """ Styler for the Back-Test Summary Table
        
#         This is the basic styler which applies some default styling to a df.

#         This shit is tedious - look at the following links if confused
#             https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
#             https://pbpython.com/styling-pandas.html
#             https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
        
#         """
        
#         # When we reset_index() the index becomes the first column
#         # by default we style that so need the column name as an indexor
#         faux_index = df.columns[0]
        
#         ## DataFrame Styler Default for Headers & Captions
#         # Sort "td" with .set_properties because it's easer to override
#         styles = [dict(selector="th",
#                        props=[("font-family", "Garamond"),
#                               ('padding', "5px 5px"),
#                               ("font-size", "15px"),
#                               ("background-color", "black"),
#                               ("color", "white"),
#                               ("text-align", "center"),
#                               ('border', '1px solid black')]),

#                   dict(selector="caption",
#                        props=[("text-align", "right"),
#                               ("caption-side", "bottom"),
#                               ("font-size", "85%"),
#                               ("color", 'grey')]),] 
        
#         df = df.style.hide_index()\
#                .set_table_styles(styles)\
#                .set_caption('Source: STANLIB Multi-Strategy')\
#                .set_table_attributes('style="border-collapse:collapse"')\
#                .set_precision(3)\
#                .highlight_null(null_color='white')\
#                .set_properties(**{"font-family": "Garamond",
#                                   "font-size": "14px",
#                                   "text-align": "center",
#                                   "border": "1px solid black",
#                                   "padding": "5px 5px",
#                                   "min-width": "70px"})\
#                .applymap(lambda x: 'color: white' if x== 0 else 'color: black')\
#                .set_properties(subset=[faux_index],
#                                **{'font-weight':'bold',
#                                   'color':'white',
#                                   'background-color':'teal',
#                                   "text-align": "justify",
#                                   'min-width':'115px'})\
                   
#         return df
    
#     def pretty_panda_summary(self):
#         """ Styler for the Back-Test Summary Table

#         This shit is tedious - look at the following links if confused
#             https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
#             https://pbpython.com/styling-pandas.html
#             https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
#         """

#         df = self.backtest_summary()
        
#         format_list = ['Payoff','profit_factor',
#                        'Beta','TE','Skew','Kurtosis','avg_drawdown_days','avg_XS_drawdown_days','recovery_factor',
#                        'tail_ratio','Sharpe','Modified_Sharpe_Ratio','IR','RaR','Calmar Ratio','Sortino','Treynor_Ratio','Omega_ratio']
# #        # Create list to format the numbesr to percentage 
# #        percentage_list = ['TR','Avg Return','avg_win','avg_loss','xs_mean','xs_worst','xs_best','Hitrate',
# #                           'Vol','Downside_Vol','TE','Historic_VaR','Historic_CVaR','Modified_VaR','Max_Drawdown','Max_XS_DD','avg_drawdown','avg_XS_drawdown']
        
        
#         # duplicate the index in the dataframe
#         # we don't just hide because we want to show & reference the index
#         m = df.index
#         m.name = 'Metric'
#         x = pd.concat([m.to_frame(), df], axis=1).fillna(0)
        
#         x = self.pretty_panda(x)
        
#         ## Generally set to 0.1%; few things as 0.02; zeros have white text
#         x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])\
#              .format(formatter="{:.2f}", subset=pd.IndexSlice[format_list, x.columns[1:]])\
             
        
#         ## Conditional Format Bits
#         # These Include the Benchmark
#         y = [['TR', 'Sharpe', 'Modified_Sharpe_Ratio','RaR', 'Max_Drawdown','Historic_VaR','Historic_CVaR','Modified_VaR'], x.columns[1:]]
#         x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
#         x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)
        
#         # These only make sense if there is more than one port being tested
#         if len(df.columns) > 2:
#             y = [['IR', 'Hitrate'], x.columns[2:]]
#             x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
#             x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)

#         return x
    
#     def pretty_panda_drawdown(self, alpha=True):
#         """ Styling for the Annualised Drawdown Tables """
        
#         # standard now - pick from dataframe based on if we want exccess rtns
#         # Sort by drawdown & pick only the last however many
#         x = self.xs_drawdown_table if alpha else self.drawdown_table
#         x = x.sort_values(by='drawdown', ascending=False).tail(10)
        
#         # useful for indexing - the formating in pandas can't take NaT
#         # so need to find the index of potential end dates that won't have ended
#         idxna = ~x['recovery'].isna()
        
#         # general stuff
#         x = self.pretty_panda(x.reset_index())
        
#         return x.format(dict(start='{:%b-%y}', trough='{:%b-%y}', drawdown='{:.1%}'))\
#                 .format(formatter="{:%b-%y}", subset=pd.IndexSlice[x.index[idxna], ['end']])\
#                 .background_gradient('RdYlGn', subset='drawdown')
    
#     def pretty_panda_annual(self, key='rtn'):
#         """ Styling for Tables of Annualised Metrics
        
#         These are calc'd by self.per_annum() which is itself in self.run_backtest()
#         Tables are stored in a dict() called self.summary_pa
        
#         Each table needs stubtly different styling
        
#         INPUT:
#             key: refers to the key from self.summary_pa
#         """
        
#         pa = self.summary_pa
        
#         # subtle differences depending on if we want Rb or not
#         # also subtle difference in if is .2% of .2f
#         if key in ['rtn']:            
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn', subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['vol']:            
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn_r', subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['downside_vol']:            
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn_r', subset=pd.IndexSlice[:, x.columns[1:]],)        
#         elif key in ['sharpe']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['alpha']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
#             x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn',  vmin=-0.05, vmax=+0.05, subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['te']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
#             x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn_r', vmin=0, vmax=0.06, subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['ir']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,3:].reset_index())
#             x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
#         elif key in ['sortino']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['Beta']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
#             x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn', vmin=-3, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
#         elif key in ['treynor']:
#             x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
#             x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
#             x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
            
#         return x
    
#     def markdown_doc(self, title="TEST"):
#         """ Master Markdown file for full backtest report """
            
#         md = []     # dummy list container - convert to strings later
    
#         # Title
#         md.append("# STANLIB Multi-Strategy Backtest")
#         md.append("### Report: {}".format(title))
#         md.append("Returns based backtest comparing portfolio(s) against the \
#                   {} benchmark; risk-free return are proxied by the {} index. \
#                   Data contains {} monthly observations running from \
#                   {:%b-%y} to {:%b-%y}. \
#                   \n \n".format(self.Rb, self.Rf,
#                                 len(self.rtns.index),
#                                 self.rtns.index[0],
#                                 self.rtns.index[-1],))
        
#         md.append("## Summary")
#         md.append(self.pretty_panda_summary().render())
#         md.append("Annualised 'risk-free' return of the {} index over the \
#                   period was {:.2%}. \n \n".format(self.Rf, self.Rf_obs_rtn))
        
#         ## Risk & Return
#         md.append("## Portfolio Returns")
#         md.append(self.plots['tr'])
#         md.append(self.plots['xsrtn'])
#         md.append(self.plots['roll_rtn'])
#         md.append(self.pretty_panda_annual('rtn').render())
#         md.append("\n \n ")
#         md.append(self.plots['roll_xsrtn'])
#         md.append(self.pretty_panda_annual('alpha').render())
        
#         ## Portfolio Risk & Drawdown
#         md.append("## Portfolio Risk & Drawdowns")
#         md.append(self.plots['roll_vol'])
#         md.append(self.pretty_panda_annual('vol').render())
#         md.append("\n \n ")
#         md.append(self.plots['roll_downside_vol'])
#         md.append(self.pretty_panda_annual('downside_vol').render())        
        
               
#         md.append(self.plots['roll_te'])
#         md.append(self.pretty_panda_annual('te').render())
#         md.append("\n \n ")
#         md.append(self.plots['xs_drawdown'])
#         md.append(self.pretty_panda_drawdown(alpha=True).render())
#         md.append(self.plots['drawdown'])
#         md.append(self.pretty_panda_drawdown(alpha=False).render())
        
#         ## Rolling Risk Adjusted Measures
#         md.append("## Risk Adjusted Returns - Rolling")
#         md.append(self.plots['roll_sharpe'])
#         md.append(self.pretty_panda_annual('sharpe').render())
#         md.append("\n \n ")
#         md.append(self.plots['roll_ir'])
#         md.append(self.plots['roll_rar'])
#         md.append(self.plots['beta'])
#         md.append(self.pretty_panda_annual('Beta').render())
#         md.append(self.plots['sortino'])
#         md.append(self.pretty_panda_annual('sortino').render())
#         md.append(self.plots['treynor'])
#         md.append(self.pretty_panda_annual('treynor').render())
        
#         ## Rolling Risk Adjusted Measures
#         md.append("## Tail-Risk - Rolling")
#         md.append(self.plots['VaR'])
#         md.append(self.plots['CVaR'])
        
#         ## Regression & Return Distributions
#         md.append("## Return Distribution")
#         md.append(self.plots['kde_rtns'])
#         md.append(self.plots['kde_alpha'])
#         md.append(self.plots['histogram'])
#         md.append("Visualising return or alpha regressions adds colour to CAPM Beta. \
#                   Steeper regression lines indicate higher Beta whilst R<sup>2</sup> gives \
#                   an impression of the correlation; look for non-linearity \
#                   that may be missed in headline metrics.")
#         md.append(self.plots['regression_rtn'])
#         md.append(self.plots['regression_alpha'])
              
#         # Hitrate
#         md.append("## Hit Rate Analysis")
#         md.append("Here we aren't interested in the quantum of return, \
#                    simply the binary outcome per month. Heatmaps will show \
#                    month-by-month experience as either +1 or 0. \
#                    For annualised analysis we look at the percentage monthly hit-rate \
#                    over a calendar year; subject to a minimum of 3-observations. \n \n")
        
#         md.append(self.plots['hitrate']['annual'].render())
        
#         for p in self.plots['hitrate']:
#             if p == 'annual':
#                 continue
#             md.append(self.plots['hitrate'][p])      
        
#         # Correlation Analysis
#         md.append("## Correlation Review")
#         md.append("We present the correlation matrix for the full sample period, \
#                    showing both the Portfolio returns and the Alpha stream. \
#                    Additionally we include a series of strategic asset classes \
#                    relevant for multi-asset portfolios. \n ")
#         md.append(self.plots['correl_wide'])
#         md.append(self.plots['correl_animation'])
#         md.append("\n \n")
        
#         return "\n \n".join(md)




# %%

class BacktestMarkdwonReport(object):
    """ Class exists for our HTML / Markdown Backtest Reports
    
    This was originally integrated into the BacktestAnalytics class but given 
    the specific edits for building an HTML report get quite long and cumbersome
    we are splitting into this seperate class.
    
    
    """    
    
    def __init__(self, backtest):
        self.bt = backtest
        return
    
    def plot_master(self, plotly2html=True, plotlyjs='cdn',
                    plot_height=450, plot_width=850):
        """ Aggregation Function that runs ALL plots
        
        These are saved in a big dictionary self.plots
        
        INPUTS:
            all related to if we want to save as HTML for output
            this is required for markdown but less useful if used in an app
        """
        
        plots = dict()   # dummy dictionary to hold plots
        
        # Total Return & Excess Return
        plots['tr'] = self.bt.plot_index(self.bt.cum_rtn,
                                      title='Cumulative Returns',
                                      ytitle='Index Level', 
                                      risk_free=True,
                                      source=True, y_src=-0.125)
        
        plots['xsrtn'] = self.bt.plot_index(self.bt.cum_xs_rtn,
                                      title='Excess Returns',
                                      ytitle='Excess Returns',
                                      benchmark=False,
                                      source=True, y_src=-0.125)
        
        # Return Distributions
        plots['kde_rtns'] = self.bt.plot_ridgeline(
                                    self.bt.rtns,
                                    title='Ridgeline KDE Distributions: Returns',
                                    source=True, y_src=-0.125)
        
        plots['kde_alpha'] = self.bt.plot_ridgeline(
                                    self.bt.xsrtns.iloc[:, 1:],
                                    title='Ridgeline KDE Distributions: Excess Returns',
                                    source=True, y_src=-0.125)
        
        # Regression Charts
        plots['regression_rtn'] = self.bt.plot_regression(
                                            alpha=False,
                                            title='Return Regression: Port Returns',
                                            source=True, y_src=-0.125)
      
        plots['regression_alpha'] = self.bt.plot_regression(
                                            alpha=True,
                                            title='Return Regression: Excess Returns',
                                            source=True, y_src=-0.125)
        
        plots['histogram'] = self.bt.plot_histo(self.bt.xsrtns,
                                    title='Excess Return Distribution',
                                    source=True, y_src=-0.125)
               
        # Drawdown Charts
        plots['drawdown'] = self.bt.plot_index(self.bt.drawdown,
                            title='Drawdown of Returns',
                            yfmt=['.1%', '.2%'], ytitle='Drawdown',
                            benchmark=True,
                            source=True, y_src=-0.125)
        
        plots['xs_drawdown'] = self.bt.plot_index(self.bt.xs_drawdown,
                            title='Drawdown of Excess Returns',
                            yfmt=['.1%', '.2%'], ytitle='Drawdown',
                            benchmark=False,
                            source=True, y_src=-0.125)
        
        # Rolling Plots
        # Rolling Period Charts
        plots['roll_rtn'] = self.bt.plot_index(self.bt.rolling[12]['rtn'],
                                            title='Rolling Return: 12m',
                                            yfmt=['.0%', '.2%'],
                                            ytitle='Return',
                                            height=350,
                                            source=True, y_src=-0.15)
        
        plots['roll_xsrtn'] = self.bt.plot_index(self.bt.rolling[12]['xsrtn'],
                                            title='Rolling Excess Return: 12m',
                                            yfmt=['.0%', '.2%'],
                                            ytitle='Alpha',
                                            benchmark=False, height=350,
                                            source=True, y_src=-0.15)
        
        plots['roll_vol'] = self.bt.plot_index(self.bt.rolling[12]['vol'],
                                            title='Rolling Volatility: 12m',
                                            yfmt=['.0%', '.2%'],
                                            ytitle='Volatility',
                                            height=350,
                                            source=True, y_src=-0.15)
        
        plots['roll_downside_vol'] = self.bt.plot_index(self.bt.rolling[12]['downside_vol'],
                                            title='Rolling Downside Volatility: 12m',
                                            yfmt=['.0%', '.2%'],
                                            ytitle='Downside Volatility',
                                            height=350,
                                            source=True, y_src=-0.15)        
        
        plots['roll_te'] = self.bt.plot_index(
                                  self.bt.rolling[12]['te'],
                                  title='Rolling ex-Post TE: 12m',
                                  yfmt=['.1%', '.2%'], ytitle='Tracking Error',
                                  benchmark=False, height=350,
                                  source=True, y_src=-0.15)

        plots['roll_sharpe'] = self.bt.plot_index(
                                  self.bt.rolling[12]['sharpe'],
                                  title='Sharpe Ratio: 12m',
                                  yfmt=['.1f', '.2f'], ytitle='Sharpe Ratio',
                                  benchmark=False, height=350,
                                  source=True, y_src=-0.15)
        
        plots['sortino'] = self.bt.plot_index(
                                self.bt.rolling[12]['sortino'],              
                                title='Rolling Sortino: 12m',
                                yfmt=['.1f', '.2f'], ytitle='Sortino',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)
        
        plots['roll_rar'] = self.bt.plot_index(
                                  self.bt.rolling[12]['xsrtn'] / self.bt.rolling[12]['vol'],
                                  title='Risk Adjusted Return: 12m',
                                  yfmt=['.1f', '.2f'], ytitle='Information Ratio',
                                  benchmark=False, height=350,
                                  source=True, y_src=-0.15)

        plots['roll_ir'] = self.bt.plot_index(
                                self.bt.rolling[12]['xsrtn'] / self.bt.rolling[12]['te'],              
                                title='Rolling Information Ratio: 12m',
                                yfmt=['.1f', '.2f'], ytitle='IR',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)
        
        plots['beta'] = self.bt.plot_index(
                                self.bt.rolling[12]['beta'],              
                                title='Rolling Beta: 12m',
                                yfmt=['.1f', '.2f'], ytitle='Beta',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)
        
        plots['treynor'] = self.bt.plot_index(
                                self.bt.rolling[12]['treynor'],              
                                title='Rolling Treynor Ratio: 12m',
                                yfmt=['.1f', '.2f'], ytitle='Treynor',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)
        
        
        plots['VaR'] = self.bt.plot_index(
                                self.bt.rolling[12]['VaR'],              
                                title='Rolling Value-at-Risk: 12m',
                                yfmt=['.1f', '.2f'], ytitle='VaR',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)   
            
        plots['CVaR'] = self.bt.plot_index(
                                self.bt.rolling[12]['CVaR'],              
                                title='Rolling Expected Shortfall: 12m',
                                yfmt=['.1f', '.2f'], ytitle='CVaR',
                                benchmark=False, height=350,
                                source=True, y_src=-0.15)   
        
        # Correlation
        plots['correl_wide'] = self.bt.plot_correl(self.bt.corr)
        plots['correl_animation'] = self.bt.plot_correl_animation(subset=True, years=1)
        
        # Convert to HTML
        if plotly2html:
            for k, v in plots.items():
                plots[k] = v.to_html(full_html=False,
                                     include_plotlyjs=plotlyjs,
                                     default_height=plot_height,
                                     default_width=plot_width,
                                     )   
        
        ### THESE USE FUNCTIONS FROM THIS CLASS NOT FROM BACKTEST CLASS
        
        # Hitrate Plots
        # These come after we've already converted most to plotly
        # that is because this is an embedded dictionary & we've already
        # converted the plotly
        plots['hitrate'] = self.plot_hitrates(show=False,
                                              plotly2html=plotly2html,
                                              plotlyjs=plotlyjs,
                                              plot_height=300,
                                              plot_width=750)
        
        self.plots = plots
        
        return plots

    def plot_hitrates(self,
                      min_count=3,
                      show=False,
                      plotly2html=True, plotlyjs='cdn',
                      plot_height=450, plot_width=850):
        
        """ Combined Function Charts & Annual Table
        
        Charts are the Year x Month, binary, hit-or-miss heatmap and
        Table is the annualised hit rate per year - in a styled dataframe
        
        IMPORTANT:
            requires bactest functions have to be run because needs
            self.xsrtns & self.rtns dataframes to exist
            uses self.plot_hitrates() function
        
        INPUTS:
            min_count: 3(default) is min. months req to get a per-year number
            
        OUTPUT:
            dict() with key 'annual' for styled df & others will be port name
            i.e. {'annual':styled_df, 'PORT1':plotly_fig}
        
        """
        
        plots = dict()        # dummy dictionary for plotly & dataframe
        df = pd.DataFrame()   # dummy dataframe for table
        
        # iterate through each portfolios alpha
        # could be done as a matrix but complexity isn't worth the speed
        # remember to remove risk-free column zero
        for i, p in enumerate(self.bt.xsrtns.iloc[:,1:]):
            
            if i == 0:
                continue
            
            ## PLOTLY
            # Get the Year x Month Hitrate Plot
            plots[p] = self.bt.plot_hitrate(self.bt.xsrtns[p],
                                          title="Hit Rate Heatmap: {}".format(p),
                                          binary=True,
                                          source=False, y_src=-0.15)
            
            ## TABLE
            # Calc the average annual across backtest things
            # use crosstab again to break down by year and month
            # then we can sum across the months
            ix = self.bt.xsrtns.loc[:, p]
            ix = pd.crosstab(index=ix.index.year,
                              columns=ix.index.strftime("%b"),
                              values=ix.values,
                              aggfunc='sum',
                              rownames=['years'],
                              colnames=['months'])
    
            # Map excess returns to hit or miss
            ix = ix.applymap(lambda x: 1 if x >= 0 else x)
            ix = ix.applymap(lambda x: -1 if x <= 0 else x)
    
            # Hit rate by month & rename result series
            ihr = ix[ix==1].sum(axis=1) / ix.count(axis=1)
            
            # May not want to distort data if we don't have a minimum no obs
            # by default we use 3-months
            if min_count > 0:
                ihr[~(ix.count(axis=1) >= min_count)] = np.nan
            ihr.name = p
    
            df = pd.concat([df, ihr.to_frame()], axis=1)
        
        if show:
            for k in plots:
                plots[k].show()
        
        if plotly2html:
            for k, v in plots.items():
                plots[k] = v.to_html(full_html=False,
                                      include_plotlyjs=plotlyjs,
                                      default_height=plot_height,
                                      default_width=plot_width,
                                      )
                  
        plots['annual'] = self.pretty_panda(df.reset_index())\
                .format(formatter="{:.0f}", subset=pd.IndexSlice[:, df.columns[0]])\
                .format(formatter="{:.1%}", subset=pd.IndexSlice[:, df.columns[0:]])\
                .background_gradient('RdYlGn', vmin=0.2, vmax=0.8, subset=pd.IndexSlice[:, df.columns[0:]])\
                .highlight_null(null_color='white')\
        
        return plots


    def pretty_panda(self, df):
        """ Styler for the Back-Test Summary Table
        
        This is the basic styler which applies some default styling to a df.

        This shit is tedious - look at the following links if confused
            https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
            https://pbpython.com/styling-pandas.html
            https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
        
        """
        
        # When we reset_index() the index becomes the first column
        # by default we style that so need the column name as an indexor
        faux_index = df.columns[0]
        
        ## DataFrame Styler Default for Headers & Captions
        # Sort "td" with .set_properties because it's easer to override
        styles = [dict(selector="th",
                       props=[("font-family", "Garamond"),
                              ('padding', "5px 5px"),
                              ("font-size", "15px"),
                              ("background-color", "black"),
                              ("color", "white"),
                              ("text-align", "center"),
                              ('border', '1px solid black')]),

                  dict(selector="caption",
                       props=[("text-align", "right"),
                              ("caption-side", "bottom"),
                              ("font-size", "85%"),
                              ("color", 'grey')]),] 
        
        df = df.style.hide_index()\
               .set_table_styles(styles)\
               .set_caption('Source: STANLIB Multi-Strategy')\
               .set_table_attributes('style="border-collapse:collapse"')\
               .set_precision(3)\
               .highlight_null(null_color='white')\
               .set_properties(**{"font-family": "Garamond",
                                  "font-size": "14px",
                                  "text-align": "center",
                                  "border": "1px solid black",
                                  "padding": "5px 5px",
                                  "min-width": "70px"})\
               .applymap(lambda x: 'color: white' if x== 0 else 'color: black')\
               .set_properties(subset=[faux_index],
                               **{'font-weight':'bold',
                                  'color':'white',
                                  'background-color':'teal',
                                  "text-align": "justify",
                                  'min-width':'115px'})\
                   
        return df
    
    def pretty_panda_summary(self):
        """ Styler for the Back-Test Summary Table

        This shit is tedious - look at the following links if confused
            https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
            https://pbpython.com/styling-pandas.html
            https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
        """

        df = self.bt.backtest_summary()
        
        format_list = ['Payoff','profit_factor',
                       'Beta','TE','Skew','Kurtosis','avg_drawdown_days','avg_XS_drawdown_days','recovery_factor',
                       'tail_ratio','Sharpe','Modified_Sharpe_Ratio','IR','RaR','Calmar Ratio','Sortino','Treynor_Ratio','Omega_ratio']
#        # Create list to format the numbesr to percentage 
#        percentage_list = ['TR','Avg Return','avg_win','avg_loss','xs_mean','xs_worst','xs_best','Hitrate',
#                           'Vol','Downside_Vol','TE','Historic_VaR','Historic_CVaR','Modified_VaR','Max_Drawdown','Max_XS_DD','avg_drawdown','avg_XS_drawdown']
        
        # duplicate the index in the dataframe
        # we don't just hide because we want to show & reference the index
        m = df.index
        m.name = 'Metric'
        x = pd.concat([m.to_frame(), df], axis=1).fillna(0)
        
        x = self.pretty_panda(x)
        
        ## Generally set to 0.1%; few things as 0.02; zeros have white text
        x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])\
             .format(formatter="{:.2f}", subset=pd.IndexSlice[format_list, x.columns[1:]])\
             
        
        ## Conditional Format Bits
        # These Include the Benchmark
        y = [['TR', 'Sharpe', 'Modified_Sharpe_Ratio','RaR', 'Max_Drawdown','Historic_VaR','Historic_CVaR','Modified_VaR'], x.columns[1:]]
        x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
        x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)
        
        # These only make sense if there is more than one port being tested
        if len(df.columns) > 2:
            y = [['IR', 'Hitrate'], x.columns[2:]]
            x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
            x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)

        return x
    
    def pretty_panda_drawdown(self, alpha=True):
        """ Styling for the Annualised Drawdown Tables """
        
        # standard now - pick from dataframe based on if we want exccess rtns
        # Sort by drawdown & pick only the last however many
        x = self.bt.xs_drawdown_table if alpha else self.bt.drawdown_table
        x = x.sort_values(by='drawdown', ascending=False).tail(10)
        
        # useful for indexing - the formating in pandas can't take NaT
        # so need to find the index of potential end dates that won't have ended
        idxna = ~x['recovery'].isna()
        
        # general stuff
        x = self.pretty_panda(x.reset_index())
        
        return x.format(dict(start='{:%b-%y}', trough='{:%b-%y}', drawdown='{:.1%}'))\
                .format(formatter="{:%b-%y}", subset=pd.IndexSlice[x.index[idxna], ['end']])\
                .background_gradient('RdYlGn', subset='drawdown')
    
    def pretty_panda_annual(self, key='rtn'):
        """ Styling for Tables of Annualised Metrics
        
        These are calc'd by self.per_annum() which is itself in self.run_backtest()
        Tables are stored in a dict() called self.summary_pa
        
        Each table needs stubtly different styling
        
        INPUT:
            key: refers to the key from self.summary_pa
        """
        
        pa = self.bt.summary_pa
        
        # subtle differences depending on if we want Rb or not
        # also subtle difference in if is .2% of .2f
        if key in ['rtn']:            
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn', subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['vol']:            
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn_r', subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['downside_vol']:            
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn_r', subset=pd.IndexSlice[:, x.columns[1:]],)        
        elif key in ['sharpe']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['alpha']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
            x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn',  vmin=-0.05, vmax=+0.05, subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['te']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
            x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn_r', vmin=0, vmax=0.06, subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['ir']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,3:].reset_index())
            x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
        elif key in ['sortino']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['Beta']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,2:].reset_index())
            x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn', vmin=-3, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
        elif key in ['treynor']:
            x = self.pretty_panda(pa[key].dropna().iloc[:,1:].reset_index())
            x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[:, x.columns[1:]])
            x = x.background_gradient('RdYlGn', vmin=-2, vmax=+3, subset=pd.IndexSlice[:, x.columns[1:]],)
            
        return x
    
    def markdown_doc(self, title="TEST"):
        """ Master Markdown file for full backtest report """
            
        md = []     # dummy list container - convert to strings later
    
        # Title
        md.append("# STANLIB Multi-Strategy Backtest")
        md.append("### Report: {}".format(title))
        md.append("Returns based backtest comparing portfolio(s) against the \
                  {} benchmark; risk-free return are proxied by the {} index. \
                  Data contains {} monthly observations running from \
                  {:%b-%y} to {:%b-%y}. \
                  \n \n".format(self.bt.Rb, self.bt.Rf,
                                len(self.bt.rtns.index),
                                self.bt.rtns.index[0],
                                self.bt.rtns.index[-1],))
        
        md.append("## Summary")
        md.append(self.pretty_panda_summary().render())
        md.append("Annualised 'risk-free' return of the {} index over the \
                  period was {:.2%}. \n \n".format(self.bt.Rf, self.bt.Rf_obs_rtn))
        
        ## Risk & Return
        md.append("## Portfolio Returns")
        md.append(self.plots['tr'])
        md.append(self.plots['xsrtn'])
        md.append(self.plots['roll_rtn'])
        md.append(self.pretty_panda_annual('rtn').render())
        md.append("\n \n ")
        md.append(self.plots['roll_xsrtn'])
        md.append(self.pretty_panda_annual('alpha').render())
        
        ## Portfolio Risk & Drawdown
        md.append("## Portfolio Risk & Drawdowns")
        md.append(self.plots['roll_vol'])
        md.append(self.pretty_panda_annual('vol').render())
        md.append("\n \n ")
        md.append(self.plots['roll_downside_vol'])
        md.append(self.pretty_panda_annual('downside_vol').render())        
        
               
        md.append(self.plots['roll_te'])
        md.append(self.pretty_panda_annual('te').render())
        md.append("\n \n ")
        md.append(self.plots['xs_drawdown'])
        md.append(self.pretty_panda_drawdown(alpha=True).render())
        md.append(self.plots['drawdown'])
        md.append(self.pretty_panda_drawdown(alpha=False).render())
        
        ## Rolling Risk Adjusted Measures
        md.append("## Risk Adjusted Returns - Rolling")
        md.append(self.plots['roll_sharpe'])
        md.append(self.pretty_panda_annual('sharpe').render())
        md.append("\n \n ")
        md.append(self.plots['roll_ir'])
        md.append(self.plots['roll_rar'])
        md.append(self.plots['beta'])
        md.append(self.pretty_panda_annual('Beta').render())
        md.append(self.plots['sortino'])
        md.append(self.pretty_panda_annual('sortino').render())
        md.append(self.plots['treynor'])
        md.append(self.pretty_panda_annual('treynor').render())
        
        ## Rolling Risk Adjusted Measures
        md.append("## Tail-Risk - Rolling")
        md.append(self.plots['VaR'])
        md.append(self.plots['CVaR'])
        
        ## Regression & Return Distributions
        md.append("## Return Distribution")
        md.append(self.plots['kde_rtns'])
        md.append(self.plots['kde_alpha'])
        md.append(self.plots['histogram'])
        md.append("Visualising return or alpha regressions adds colour to CAPM Beta. \
                  Steeper regression lines indicate higher Beta whilst R<sup>2</sup> gives \
                  an impression of the correlation; look for non-linearity \
                  that may be missed in headline metrics.")
        md.append(self.plots['regression_rtn'])
        md.append(self.plots['regression_alpha'])
              
        # Hitrate
        md.append("## Hit Rate Analysis")
        md.append("Here we aren't interested in the quantum of return, \
                   simply the binary outcome per month. Heatmaps will show \
                   month-by-month experience as either +1 or 0. \
                   For annualised analysis we look at the percentage monthly hit-rate \
                   over a calendar year; subject to a minimum of 3-observations. \n \n")
        
        md.append(self.plots['hitrate']['annual'].render())
        
        for p in self.plots['hitrate']:
            if p == 'annual':
                continue
            md.append(self.plots['hitrate'][p])      
        
        # Correlation Analysis
        md.append("## Correlation Review")
        md.append("We present the correlation matrix for the full sample period, \
                   showing both the Portfolio returns and the Alpha stream. \
                   Additionally we include a series of strategic asset classes \
                   relevant for multi-asset portfolios. \n ")
        md.append(self.plots['correl_wide'])
        md.append(self.plots['correl_animation'])
        md.append("\n \n")
        
        return "\n \n".join(md)

# %% TEST CODE
  
def test_code():
    """ Code uses BACKTEST.xlsm which has timeseries data in it """
    
    import xlwings as xlw
    import pandas as pd
    from topgun.reporting import Reporting

    wb = xlw.Book('BACKTEST.xlsm')

    # index data from timeseries sheet
    benchmarks = wb.sheets['TIMESERIES'].range('D1').options(pd.DataFrame, expand='table').value.iloc[3:,:]
    benchmarks.index = pd.to_datetime(benchmarks.index)
    
    # Extract the Core and the Enhanced Strategies from the ports
    # then splice to get a return series
    E = wb.sheets['Enhanced'].range('A1').options(pd.DataFrame, expand='table').value.iloc[:,1]
    C = wb.sheets['Core'].range('A1').options(pd.DataFrame, expand='table').value.iloc[:,1]
    E.index = E.index + pd.offsets.MonthEnd(0)
    C.index = C.index + pd.offsets.MonthEnd(0)
    E.name = 'Enhanced'
    C.name = 'Core'
    rtns = pd.concat([E, C], axis=1).dropna()    # strategy returns
    
    # Add a blended portfolio that is a mix of strategy E & C
    x = 0.5
    rtns['E50'] = rtns['Enhanced'] * x + rtns['Core'] * (1 - x)
    
    ### NOW WE TEST
    bt = BacktestAnalytics(rtns, benchmarks, bmks_as_rtns=False, benchmark='SWIX', Rf='STEFI')
    
    #bt.run_backtest()
    #bt.plot_master()
    bt.big_bang()
    
    # produce report
    Reporting().md2html(md=bt.markdown_report, title='test MM1')
    
    return bt

bt = test_code()