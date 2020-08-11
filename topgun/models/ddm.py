# -*- coding: utf-8 -*-
"""
Dividend Discount Model(s)

Series of functions for solving dividend discount model problems. Primarily

- dividend_discount_model_irr() is a sovler for given px, divident stream & G
- multi_stage_irr() is a 3-stage model which estimates g = ROE * (1-PO)

"""

# %% REQUIRED PACKAGES

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# %% Class Module Test

class dividend_discount_models(object):
    """ 
    
    
    """
    
    def __init__(self, **kwargs):
        
        # Test thing
        self.test = "badgers"
        
        # Kwargs & General Settings
        self.data = kwargs['data'] if 'data' in kwargs else None
        self.data_gdp = kwargs['data_gdp'] if 'data_gdp' in kwargs else None
        self.eom = kwargs['eom'] if 'eom' in kwargs else True
        
        return
            
    # %% Admin & Data Srubbing
    
    # Collect Equity Metrics 2 Single Dataframe
    def data_from_ticker(self, ticker, dd=None, eom=True):
        """ Collect Equity Metrics (PX, PE etc...) 2 single DF
        
        requires input dictionary of form {'PX':pe_dataframe}
        where:
            pe_dataframe needs the ticker in the column varnames  
        
        if eom == True all dates will be set to End-of-Month
        """
        
        # pull from internal if data dictionary not supplied
        dd = self.data if dd == None else dd
        
        # iterate through each key in dict of field-dataframes
        df = pd.DataFrame(columns=dd.keys())     # Dummy Dataframe                  
        for i, key in enumerate(dd.keys()):
        
            ix = dd[key][ticker]
            if eom:
                ix.index = ix.index + pd.tseries.offsets.MonthEnd(0)                 
            df[key] = ix    # populate dataframe
        
        self.ticker_data = df
        return df
    
    # Terminal Growth Trends
    def terminal_gdp_per_capita(self, gdp=None, w=10, smoothing=3, pull2us=0):
        """ Calculate Terminal Growth Rates from GDP Per Capita """
        
        # use class attribute if available (and None specified)        
        gdp = self.data_gdp if gdp == None else gdp
        
        # Calculate geo mean of Real GDP per capita & smooth if req.
        x = (1 + gdp.pct_change(1)).rolling(w).apply(np.prod).apply(lambda x: np.power(x, 1/w))-1
        x = x.rolling(smoothing).mean()  # smoothing if required
        
        # Form quick EM Composite
        wgtCNY = 0.4    # China Weight
        x['EM'] = (1-wgtCNY)*x.loc[:,['BRL','RUB','INR','MXN','ZAR']].mean(axis=1) + x['CNY']*wgtCNY
        
        # pull-to-US if specified
        if pull2us > 0:
            for c in x:
                x.loc[:,c] = (x.loc[:, c]*(1-pull2us)) + ((pull2us * x.loc[:,'USD']))
        
        self.terminal_G = x
        return x
    
    def _scrub_field_data(self, lim_dy=[0, 100], lim_roe=0, winsorize_lim=[0.05, 0.05]):
        
        from scipy.stats.mstats import winsorize
        
        # ensure all data are floats
        # Replace None with np.nan & set all data to floats (maybe more later)
        for k in self.data.keys():
            self.data[k] = self.data[k].replace('None', np.nan).astype(float)
        
        # Limit Dividend Yields to 0-100%
        DY = self.data['DY']
        DY[DY < lim_dy[0]] = lim_dy[0]
        DY[DY > lim_dy[1]] = lim_dy[1]
        self.data['DY'] = DY
        
        # Floor ROE at 0
        ROE = self.data['ROE']
        ROE[ROE < lim_roe] = lim_roe
        self.data['ROE'] = ROE
        
        # Winsorize if available
        for k in ['ROE', 'PE', 'FwdPE']:
            # first check if field is a key in the data dict
            if k in self.data.keys():
                
                # then iterate over columns winsorising (doesn't work in 2D)
                x = self.data[k]
                for t in x:
                    x[t] = winsorize(x[t], winsorize_lim)
                self.data[k] = x
            
        return self.data

    # %% DDM Solver Functions - GIVEN DIVIDEND STREAM
    # Build a dividend stream using another function - solve it here
    
    def ddm_irr(self, px, vDividend, G):
        """ IRR finder for Dividend Discount Model in SciPy
        
        Uses the scipy.optimise minimize_scalar optimisation function
            minimize_scalar(f, args=(d, px, G)).x
            
        Where d - vector of dividend stream from d0 to dTerminal
              px - current index level associated with dividend stream
              G = perpetual growth rate
        """
        x = minimize_scalar(self._solver_ddm_irr, args=(px, vDividend, G)).x
        return x
    
    def _solver_ddm_irr(self, x, px, d, G):
        """ solver function for dividend_discount_model_irr"""
        pv = 0     # Set present value of div stream to 0 to t0
        for i, v in enumerate(d):
            
            if i == 0:              # skip d0 which isn't included in DDM calc
                continue
            elif i < len(d) - 1:    # len(d)-1 == terminal year
                pv += v / ((1 + x) ** i)
            else:
                pv += (v / ((1 + x) ** i)) * ( 1 / (x - G))
                
        return np.abs(px - pv)    # minimise in optimiser
    
    # %% Sustainable Return Model
    
    # Multi-Ticker Latest
    def sustainable_rtn_multi(self, ticker_list=['MXUS'],
                              roe_trend=120, po_trend=120,
                              trend_start=10, terminal=21,
                              fwd_pe = False):
        
        """ Table of Most Recent Observations for MULTI tickers
        
        NB/ This is NOT an efficient method for doing this FURTHER WORK """
        
        ### MOVE THIS ELSEWHERE
        mapping = dict(MXWO = {'toto':'World', 'fx':'USD'},
                       MXEF = {'toto':'World', 'fx':'EM'},
                       MXUS = {'toto':'United States', 'fx':'USD'},
                       MXEA = {'toto':'', 'fx':'USD'},
                       MXGB = {'toto':'United Kingdom', 'fx':'GBP'},
                       MXEUG = {'toto':'Europe', 'fx':'EUR'},
                       MXEU = {'toto':'Europe', 'fx':'EUR'},
                       MXZA = {'toto':'South Africa', 'fx':'ZAR'},
                       MXJP = {'toto':'Japan', 'fx':'JPY'},
                       SPX = {'toto':'United States', 'fx':'USD'},
                       UKX = {'toto':'United Kingdom', 'fx':'GBP'},
                       JALSH = {'toto':'South Africa', 'fx':'ZAR'},
                       SXXP = {'toto':'Europe', 'fx':'EUR'},
                       TPX = {'toto':'Japan', 'fx':'JPY'},) 
        
        # Dummy Dataframes
        ExRtn, ERP = [pd.DataFrame(columns=ticker_list)] * 2
        vn = ['PX','ExRtn','DY','PE','ROE','PO','g','G','ROE_trend','PO_trend']
        t0 = pd.DataFrame(index=ticker_list, columns=vn)
        
        # iterate through each ticker
        for t in ticker_list:
            
            fx = mapping[t]['fx']
            df = self.data_from_ticker(t)
            ddm, _ = self.sustainable_rtn_ts(df=df, G_fx=fx,
                                             roe_trend=roe_trend,
                                             po_trend=po_trend,
                                             trend_start=trend_start,
                                             terminal=terminal,
                                             fwd_pe = False)
            
            #populate ts ExRtns & ERP as well as t0 current input data
            ExRtn.loc[:, t] = ddm['ExRtn']
            vn = ['PX','ExRtn','DY','PE','ROE','PO','g','G','ROE_trend','PO_trend']
            t0.loc[t, vn] = ddm.loc[ddm.index[-1], vn]
                
        return t0, ExRtn
        
    # Full timeseries for SINGLE Ticker
    def sustainable_rtn_ts(self, df=None, G_fx='USD',
                           roe_trend=120, po_trend=120,
                           trend_start=10, terminal=21,
                           fwd_pe = False,):
        
        """ Timeseries of Sustainable Returns for Single Ticker

        Sustainable Return model using current PX, ROE & Payout-Ratios. Where
        we estimate g = ROE * (1-PO). Currently 3 steps:
            1. converge to trend
            2. time in trend
            3. terminal growth rate
            
        INPUTS:
            df = dataframe for 1 name, with fields as column names
            G_fx = given as FX code. REQUIRES terminal_G df within class
            trend (in months) for ROE & Payout Ratios
            trend & terminal years.         
        
        """
        
        # determine input dataframe to use - order of if's important here
        if type(df) == str:
            # if a string provided assumes its a ticker & build from scratch
            # will require self.data dictionary to have been populated!!!
            df = self.data_from_ticker(df) 
        elif isinstance(df, pd.DataFrame):
            df = df   # THIS MAKES NO SENSE (HOW TO KILL THIS LINE?)
        else:
            # class attribute - comes 3rd or will fail if string input (above)     
            df = self.ticker_data             # class attribute

        x = df.copy()    # avoid editing original data
        
        # Update Column Name to make indexing easier later (FwdPE is optional)
        vn = ['PX', 'DY', 'PE', 'ROE', 'FwdPE']
        vn = vn[0:-1] if len(x.columns)==4 else vn    
        x.columns = vn
    
        x['E0'] = x.PX / x.PE        # PE implied Earnings (pre-Inflation adj)
        
        # Dividend Stuff
        x['DY'] = x.DY / 100      # convert to %
        x['D0'] = x.PX * x.DY        # Dividned Stream (POST inflation adj)
        x['PO'] = x.DY / (1/x.PE)    # PO = DY / EY
        x['PO_trend'] = x.PO.rolling(po_trend).median()
    
        # Include forward earnings
        if fwd_pe == True:
            x['E1'] = x.PX / x.FwdPE       # implied earnings next year
            x['POi'] = x.DY / (1/x.FwdPE)  # implied PO at t1
            
        # Growth Stuff
        x.ROE = x.ROE / 100    # convert to %
        x['ROE_trend'] = x.ROE.rolling(roe_trend).median()  # find rolling average ROE
        x['g'] = x.ROE * (1 - x.PO)                         # EPS growth from g = EPS * (1 - PO)
        x['g_trend'] = x.ROE_trend * (1 - x.PO_trend)       # same but finding trend growth
        
        # Terminal Growth Rate
        # class terminal_G table used (see admin) & "FX" code to be used
        G = self.terminal_G[G_fx]
        idx = x.index.intersection(G.index)    # find index of matching dates
        x.loc[idx, 'G'] = G[idx]               # populate Real G from table
        x['G'] = x['G'].ffill()                # ffill annual data
        
        ### Iterate through point in time calculate IRR
        # remeber the point of the above is so that we have all the data we need for the calc in one row
        x.dropna(how='any', inplace=True)   # bin rows with missing data
        
        for i, v in enumerate(x.index):
        
            # Run DDM code
            ddm, res = self.sustainable_return_calc(x.iloc[i,:],
                                                    trend_start=trend_start,
                                                    terminal=terminal)
            x.loc[v, 'ExRtn'] = res        # populate table
        
        # x is the summary table; ddm is the actual dividend calculation
        return x, ddm
    
    def sustainable_return_calc(self, dv,
                                trend_start=10,
                                terminal=21, **kwargs):
        
        """ Multi-Stage Dividend Discount Model solving for IRR
        
        Assumtion is 3 phases:
            1. Convergence to trend
            2. Time in trend
            3. Perpetual growth
    
        growth with growth estimated as g = ROE * (1 - Payout_Ratio)
        
        INPUTS:
            dv = data-vector which can either be a list or a pd.Series; must contain: 
                 ['PX', 'D0', 'ROE', 'PO', 'ROE_trend', 'PO_trend', 'G']
            trend_start = int of year (default=10) entering trend growth
            terminal = int or year (default=21) entering perpetual growth phase
        
        """
        if isinstance(dv, list):
            vn = ['PX', 'D0', 'ROE', 'PO', 'ROE_trend', 'PO_trend', 'G']
            dv = pd.Series(data=dv, index=vn)#.T    # Transpose as lists form column df
            
        # Set up output Dataframe, with the Index representing years
        ddm = pd.DataFrame(index=range(terminal+1), columns=['D','ROE','PO','g','PX'])
        
        # T0
        ddm.loc[0,['D', 'ROE', 'PO', 'PX']] = dv.D0, dv.ROE, dv.PO, dv.PX
            
        # Phase 1 - converge to trend    
        ddm.loc[0:trend_start, 'ROE'] = np.linspace(ddm.loc[0,'ROE'], dv.ROE_trend, trend_start+1).ravel()
        ddm.loc[0:trend_start, 'PO'] = np.linspace(ddm.loc[0,'PO'], dv.PO_trend, trend_start+1).ravel()
        # Phase 2 - time in trend
        ddm.loc[trend_start:, 'ROE'] = dv.ROE_trend
        ddm.loc[trend_start:, 'PO'] = dv.PO_trend
        
        # implied g, Terminal G & update dividends
        ddm['g'] = ddm.ROE * (1 - ddm.PO) 
        for i, v in enumerate(ddm.index[:-1]):
            ddm.loc[i+1, 'D'] = ddm.loc[i, 'D'] * (1 + ddm.loc[i, 'g'])
        
        ddm.loc[ddm.index[-1], 'g'] = dv['G']    # set Terminal Growth Rate in table
        
        r = self.ddm_irr(dv.PX, ddm.D, dv.G)
    
        # Show present value in table    
        for i, v in enumerate(ddm.index):
            if i == 0:
                continue
            elif v < terminal:
                ddm.loc[v,'PX'] = -1 * ddm.loc[v,'D']/((1+r)**i)
            else:
                ddm.loc[v,'PX'] = (ddm.loc[v,'D'] /((1+r)**i)) * -1 * (1 / (r - ddm.g.iloc[-1]))
            
        return ddm, r
     
 # %% TESTING
    
# import xlwings as xlw
# wb = xlw.Book('DM Chartbook.xlsm')
# pxlw = lambda a, b: wb.sheets[a].range(b).options(pd.DataFrame, expand='table').value
# px = pxlw('MSCI_PX', 'D1').iloc[3:,:]
# pe = pxlw('MSCI_PE', 'D1').iloc[3:,:]
# dy = pxlw('MSCI_DY', 'D1').iloc[3:,:]
# roe = pxlw('MSCI_ROE', 'D1').iloc[3:,:]
# gdp = pxlw('GDP_PC', 'D1').iloc[3:,:]

# dd = {'PX': px, 'DY': dy, 'PE': pe, 'ROE': roe}
# ddm = dividend_discount_models(data=dd, data_gdp=gdp)
# ddm.terminal_gdp_per_capita()
# ddm._scrub_field_data()
#x = ddm.sustainable_rtn_ts('MXWO', 'USD')
