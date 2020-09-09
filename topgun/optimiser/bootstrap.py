# -*- coding: utf-8 -*-
"""
Bootstrap - Top Gun Stochastic Modelling Class

Created on Tue Sep  8 08:17:30 2020
@author: David J McNay
"""

# %% IMPORTs CELL

# Default Imports
import numpy as np
import pandas as pd
import scipy.linalg as LA 
import xlwings as xlw        # less useful in production

# Plotly for charting
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
#from plotly.offline import plot
pio.renderers.default='browser'

# %% CLASS MODULE

class bootstrap(object):
    """ Portfolio Stochastic Modelling Class Modules
    
    MAIN FUNCTIONS:
        emperical() - runs emperical sims for 1 vector of weights
        sim_stats() - calc descriptive stats for 1 simulation output
        port_stats() - rtn, vol & marginal-contribution given inputs
        emperical_frontier() - runs emperical analysis across all ports in wgts
                    will do stochastic sims, sim_stats, port_stats & return 
                    a dictionary with all the outputs (stored as self.results)
        correl_rmt_filtered() - allows us to build RMT filtered correl matrix
                    for other correl work look at correls module
    
    CHARTING FUNCTIONS:
        
    
    INPUTS:
        wgts - dataframe where indices are asset classes & cols are portfolios
        mu - vector of expected returns (as pd.Series) 
        vol - vector of expected volatilies (as pd.Series)
        hist - dataframe of historic returns (NOT index px/levels)
        cor - dataframe of correlation matrix
        nsims - number of Monte Carlo simulations
        psims - no of periods to run simulation over (default = 260w)
        f - annualisation factor (default = 52 for weekly returns data)
        
    DEVELOPMENT:
        - check correlation matrix PSD in class properties

    Author: David J McNay
    """

    ## Initialise class
    def __init__(self,
                 wgts=None,
                 tgts=None,
                 mu=None,
                 alpha=None,
                 vol=None,
                 te=None,
                 hist=None, 
                 cor=None,
                 nsims=1000,
                 f=52,
                 psims=260,
                 **kwargs):
        
        # ORDER OF INITIALISATION IS IMPORTANT
        self.wgts = wgts
        self.tgts = tgts
        self.mu = mu          # vector check in properties
        self.alpha = alpha
        self.vol = vol        # vector check in properties
        self.te = te
        self.hist = hist
        self.cor = cor        # check symmetrical
        
        # TOOLS
        self.nsims = nsims    # number of simulations
        self.f = f            # annualisation factor
        self.psims = psims    # no of periods in MC simulation
        
        # Inputs with some re-jigging
        if cor is None and hist is not None:
            self.cor = hist.corr()                 # basic covariance matrix
            self.cor = self.correl_rmt_filtered()  # run RMT filter as default
            
        ## Update Plotly template
        colourmap = ['grey', 'teal', 'purple', 'green','grey', 'teal', 'purple', 'green','grey', 'teal', 'purple', 'green',]
        fig = go.Figure(layout=dict(
                      font={'family':'Calibri', 'size':14},
                      plot_bgcolor= 'white',
                      colorway=colourmap,
                      showlegend=True,
                      legend={'orientation':'v'},
                      margin = {'l':75, 'r':50, 'b':25, 't':50},
                      xaxis= {'anchor': 'y1', 'title': '', 'hoverformat':'.1%', 'tickformat':'.0%',
                              'showline':True, 'linecolor': 'gray',
                              'zeroline':True, 'zerolinewidth':1 , 'zerolinecolor':'whitesmoke',
                              'showgrid': True, 'gridcolor': 'whitesmoke',
                              },
                      yaxis= {'anchor': 'x1', 'title': '', 'hoverformat':'.1%', 'tickformat':'.0%',
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
                      annotations=[{'text': 'Source: STANLIB Multi-Strategy',
                                    'xref': 'paper', 'x': 0.9, 'ax': 0,
                                    'yref': 'paper', 'y': -0.05, 'ay': 0,}],))
        
        # Save template
        pio.templates['multi_strat'] = pio.to_templated(fig).layout.template
        
        return
    
    # %% CLASS PROPERTIES
        
    # Expected Returns (mu) - Ideally pd.Series but MUST be a (1xN) vector 
    @property
    def mu(self): return self.__mu
    @mu.getter
    def mu(self): return self.__mu
    @mu.setter
    def mu(self, x):
        if isinstance(x, pd.Series):
            x.name = 'mu'
        elif len(np.shape(x)) != 1:
            raise ValueError('mu input is non-vector: {} given'.format(x))  
        self.__mu = x
        
    # Volatility (vol) - Ideally pd.Series but MUST be a (1xN) vector 
    @property
    def vol(self): return self.__vol
    @vol.getter
    def vol(self): return self.__vol
    @vol.setter
    def vol(self, x):
        if isinstance(x, pd.Series):
            x.name = 'vol'
        elif len(np.shape(x)) != 1:
            raise ValueError('vol input is non-vector: {} given'.format(x))  
        self.__vol = x
        
    # Correlation Matrix
    # Currently just check if symmetrical
    # Add test positive semi-definate 
    @property
    def cor(self): return self.__cor
    @cor.getter
    def cor(self): return self.__cor
    @cor.setter
    def cor(self, x):
        if x.shape[0] != x.shape[1]:
            raise ValueError('Correl Matrix non-symmetrical: {} given'.format(x))            
        self.__cor = x
    
    
    # nsims - number of simulations to run - needs to be an integer
    @property
    def nsims(self): return self.__nsims
    @nsims.getter
    def nsims(self): return self.__nsims
    @nsims.setter
    def nsims(self, x):
        if not isinstance(x, int):
            raise ValueError('nsims needs to be an integer: {} given'.format(x)) 
        self.__nsims = int(x)
        
    # psims - number of periods per MC Sim - needs to be an integer
    @property
    def psims(self): return self.__psims
    @psims.getter
    def psims(self): return self.__psims
    @psims.setter
    def psims(self, x):
        if not isinstance(x, int):
            raise ValueError('psims needs to be an integer: {} given'.format(x)) 
        self.__psims = int(x)
    
    # f - annualisation factor needs to be an integer
    @property
    def f(self): return self.__f
    @f.getter
    def f(self): return self.__f
    @f.setter
    def f(self, x):
        if not isinstance(x, int):
            raise ValueError('annualisation factor needs to be an integer: {} given'.format(x)) 
        self.__f = int(x)

    # %% Emperical Bootstrap
    
    def emperical(self, **kwargs):
        """ Monte-Carlo Simulation using Scaled Empirical Data
        
        Jacos idea to take the historical timeseries and standardise, then once
        standardised we can input our own means and volatility estimates. This
        will maintain higher moments (skew & kurtosis) from the original ts
        but allow us to use our own forward estimates.
        
        Note a serious problem of this approach is the length of the historical
        data. Correlation is essentially taken by picking x random periods from 
        this data - as a result we are literally just recycling the same periods
        over and over in a new order making this analysis less useful for longer
        simulations or sims where this historical period is short.
        
        OUTPUT:
            pd.DataFrame with each simulation being a row (starting at 0) and 
            each column being a period along the sim. Column[0] representing
            time-0 is set at a portfolio value of 1
        
        INPUTS:
            w = vector of port wgts ideally pd.Series()
            mu = vector of exp rtns idieally pd.Series()
            alpha (OPTIONAL) = vector of asset class alpha expectations
            vol = vector of annualised volatilities
            te (OPTIONAL) tracking error of alpha sources to asset class beta
            f = int() annualisation factor (default=52)
            nsims = int() number of simulations
            psims = int() no of observation periods simulation
            
        DEVELOPMENTS:
        * Correlation of Alpha sources == 0; could incorporate alpha correl matrix
        * Converting hist_rtns to np.array may speed things up; rest is already np
            
        Author: Jaco's brainpower; adapted by David 
        """
        
        ## INPUTS
        w = kwargs['w'] if 'w' in kwargs else self.w
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu
        vol = kwargs['vol'] if 'vol' in kwargs else self.vol
        hist = kwargs['hist'] if 'hist' in kwargs else self.hist
        nsims = kwargs['nsims'] if 'nims' in kwargs else self.nsims
        f = kwargs['f'] if 'f' in kwargs else self.f
        psims = kwargs['psims'] if 'psims' in kwargs else self.psims
        
        ## OPTIONAL INPUTS
        alpha = np.zeros(len(w)) if 'alpha' not in kwargs else kwargs['alpha']
        te = np.zeros(len(w)) if 'te' not in kwargs else kwargs['te']
        
        # De-Annualise Returns & Vols
        mu, alpha = (mu / f), (alpha / f)
        vol, te   = (vol / np.sqrt(f)), (te / np.sqrt(f))
        
        # Re-scale historical return series
        std_rtn = (hist - hist.mean()) / hist.std()         # standardise
        std_rtn = std_rtn.mul(vol, axis=1).add(mu, axis=1)  # re-scale
    
        for i in range(0, nsims):
            
            #irtn = std_rtn.iloc[:simlength]
            irtn = std_rtn.sample(psims)
            ialpha = np.random.normal(alpha, te, (psims, len(w)))
            irtn = irtn + ialpha
            
            # Build simulated path & add to simulations array
            path = (1 + (irtn @ w)).cumprod(axis=0)
            
            # create sims array on 1st iteration
            # add to sims stack on further iterations
            if i == 0:
                sims = path
            else:
                sims = np.vstack((sims, path))
        
        # Convert to DataFrame - adjust columns to start at 1 not 0 
        # insert vec PortValue==1 at col.0; concat because pd.insert is crap
        df = pd.DataFrame(sims, columns=range(1, psims+1))
        v1 = pd.DataFrame(np.ones((nsims, 1)), columns=[0])
            
        return pd.concat([v1, df], axis=1)
    
    
    def sim_stats(self, sims, tgt=0, method='annualise', **kwargs):
        """ Descriptive Statistics for dataframe of Monte Carlo Sims
        
        INPUTS:
            sims - df with rows as sims; columns as periods along sim path
            tgt - numerical return bogie (default = 0)
            periods - list periods on sim path to calc (default = all > 1yr)
                note column varnames must be numerical as used in annualisation
            method   annualise (default) - annualises periodic return
                     relative - subtracts annualised return by target
                     terminal - looks at terminal value
        
        Author: David J McNay 
        """
                
        # percentiles we want to see
        pc = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
        
        # periods from simulations to analyse
        if 'periods' in kwargs:
            periods = kwargs['periods']
        else:
            periods = sims.columns[self.f:]
            
        ## SUBSET & ANNUALISE
        
        # subset sims to required dates; if len(periods) is 1 this will return
        # a pandas series so need to convert back to df (to access cols later)
        sims = sims.loc[:, periods]
        sims = sims.to_frame() if isinstance(sims, pd.Series) else sims
        
        anns = sims ** (self.f / sims.columns) - 1     # annualised rtns from paths
        
        # Alternative calc methods available
        #    0. 'annualise' (default) annualised returns
        #    1. relative reduces returns by target (& tgt == 0)
        #    2. terminal assumes portval rtns to sims (& tgt == 1)
        if method == 'relative':
            anns, tgt = (anns - tgt), 0
        elif method == 'terminal':
            anns, tgt = sims, 1
    
        # Stats (not computed by pd.describe)
        stats = pd.DataFrame(index=anns.columns)
        stats['median'] = anns.median()
        stats['skew'] = anns.skew()
        stats['kurtosis'] = anns.kurtosis()
        stats['target'] = tgt
        stats['prob'] = anns[anns > tgt].count() / anns.count()

        return pd.concat([stats.T, anns.describe(percentiles=pc)], axis=0)
    
    def port_stats(self, w=None, mu=None, vol=None, cor=None, **kwargs):
        """ Portfolio Risk & Return Stats including MCR
        
        NB/ This function ought to be elsewhere in the package; it may therefore
        get replicated and then removed in the fullness of time.
        
        INPUT:
            w - wgts df with assets in index & ports as cols
            mu - pd.Series of expected returns
            vol - pd.Series of volatilities
            cor - np.array or pd.Dataframe correlation matrix
            
        OUTPUT: dictionary with keys risk, rtn, mcr, tcr & pcr
        
        REFERENCES:
            [1] http://webuser.bus.umich.edu/ppasquar/shortpaper6.pdf
        
        Author: David (whilst loitering in the Scottish sunshine)
        """
    
        ## INPUTS - from self if None provided        
        if w is None: w = self.wgts
        if mu is None: mu = self.mu
        if vol is None: vol = self.vol
        if cor is None: cor = self.cor

        ## CALCULATIONS
        rtn = w.multiply(mu, axis=0).sum()        # expected return
        
        # convert w to dataframe if series passed
        # this is a problem with the change in dimensions
        if isinstance(w, pd.Series):
            w = w.to_frame()
            
        wa = np.array(w)                          # wgts to arrays for matrix algebra
        vcv = np.diag(vol) @ cor @ np.diag(vol)   # covariance matrix
        v = np.sqrt(np.diag(wa.T @ vcv @ wa))     # portfolio volatility
        
        # Marginal Contribution to Risk
        # where MCR = (w.T * VCV) / vol
        mcr = np.transpose((wa.T @ vcv) / v.reshape((w.shape[1],1)))
        mcr.columns, mcr.index = w.columns, mu.index
        tcr = mcr * wa    # total contibution to risk 
        pcr = tcr / v     # percentage TCR (sums to 100%)
        
        # convert vol pack to pandas series
        v = pd.Series(data=v, index=[rtn.index])
        
        # Ingest to class
        self.port_rtn = rtn
        self.port_vol = v
        self.mcr = mcr
        self.tcr = tcr
        self.pcr = pcr
        
        return dict(risk=v, rtn=rtn, mcr=mcr, tcr=tcr, pcr=pcr)
    
    def emperical_frontier(self, wgts=None, tgts=None, alpha=True, **kwargs):
        """ Runs Stochastic Modelling on whole Frontier
        
        Frontier here refers to any set of portfolio weights - original use case
        was to run analysis on each port on an MVO efficient frontier
                
        """
        
        ## INPUTS
        
        # pull wgts dataframe from self if None provided
        wgts = self.wgts if wgts is None else wgts
        
        # Return Targets can be provided, pulled from object or zeros
        if tgts is None:
            # if None provided grab tgts from self
            tgts = self.tgts    
        elif tgts == 0:
            # array of zeros otherwise
            tgts = np.zeros(wgts.shape[1])
            
        # Alpha
        # Remember that alpha & te are set ONLY via kwargs in emperical bootstrap
        # For frontier if alpha is false create 2x series of zeros
        if alpha:
            alpha, te = self.alpha, self.te
        else:
            alpha = pd.Series(name='alpha', index=wgts.index,
                              data=np.zeros(wgts.shape[0]))
            te = pd.Series(name='te', index=wgts.index, 
                           data=np.zeros(wgts.shape[0]))
        
        # Output storage dictionary
        # keys as names of ports being tested, values will be dicts themselves
        data = dict.fromkeys(wgts.columns)    
        
        # port_stats() works on whole frontier anyway to do before iteration
        portstats = self.port_stats(w=wgts)    # NB/ not part of MC sim
        
        ## iterate across frontier (columns of wgts df)
        for i, port in enumerate(wgts):
            
            # Pull inputs from self - for the bootstrap
            # Not technically required; useful to store so we know exactly
            # which inputs went into a given model
            # also append some portstats stuff (MCR, TCR, PCR) which is useful
            # although not used at all in the stochastic modelling
            df = pd.concat([wgts[port], self.mu, self.vol, alpha, te,
                           pd.Series(portstats['mcr'][port], name='mcr'),
                           pd.Series(portstats['tcr'][port], name='tcr'),
                           pd.Series(portstats['pcr'][port], name='pcr'),
                           ], axis=1)
            
            # rename wgt vector to w (rather than the port name)
            df.rename(columns={port:'w'}, inplace=True)
            
            # Run simulation. Pass wgts but take f, nsims, psims from self 
            # For emperical() alpha is set via kwarg, send zeros if alpha False
            sims = self.emperical(w=wgts[port], alpha=alpha, te=te)

            # annualised returns from sim paths
            # and descriptive stats
            annsims= sims ** (self.f / sims.columns) - 1
            simstats= self.sim_stats(sims, tgt=tgts[i])
        
            
            irisk = pd.Series([df.w.T @ df.mu,                # portfolio return
                               df.w.T @ df.alpha,             # alpha rtn
                               df.tcr.sum()],                 # vol
                               index=['port_rtn', 'alpha_rtn', 'port_vol'],
                               name=port)
        
            # Dictionary of all useful stuff for this simulation
            port_data = dict(inputs=df, 
                             sims=sims,
                             annsims=annsims,
                             stats=simstats,
                             tgt=tgts[i],
                             risk_rtn=irisk)
            
            # merge ith port dictionary with output dict
            data[port] = port_data
        
        self.results = data    # save results to self
        return data
    
    # %% Correlation Functions
    # This may be duplicated elsewhere in the package
    # Ideally we'd put it in our correlation functions group
    
    def correl_rmt_filtered(self, c=None, from_returns=False):
        """ Create Random Matrix Theory Filtered CoVariance
        
        Props to Jaco who showed me how to do this. We de-noise our input 
        correlation matrix by comparing the eigns from a PCA to the eigns
        or a randomly generated matrix. Then we scale the original matrix
        by removing and eigen vectors where the eignenval < random matrix.
        
        INPUTS:
            c - correlation matrix as dataframe (ideally) 
                None (default) reverts to self.cor
        
        Author: David J McNay (sort of) but credit goes to Jaco; Sept 2020
        """
        
        if c is None:
            c = self.cor
        
        # Use as a flag to determine if input is a dataframe or not
        # if so we convert back to dataframe later
        pandapower = True if isinstance(c, pd.DataFrame) else False
        
        # find ordered eigenvalues of input corr matrix
        w0, v0 = self._ordered_eig(c)
        
        # Generate multivariate gaussian of the same size
        # then find eigens of random returns matrix
        RANDRETS = np.random.standard_normal(size = c.shape)
        rand_cor = np.corrcoef(RANDRETS, rowvar=False)
        wR, vR = self._ordered_eig(rand_cor)
        
        #If the eigenvalue larger than the eigen from random matrix include, else set zero
        w = []
        for e0, eR in np.c_[w0, wR]:
            if e0 > eR:
                w.append(e0)
            else:
                w.append(0)
                
        D = np.diag(np.array(w))
    
        # Recover correlation matrix from filtered eigen values and original vectors
        # Set diagonals to one
        c1 = v0 @ D @ v0.T
        c1 = np.eye(c.shape[0]) - np.diag(np.diag(c1)) + c1
        
        if pandapower:
            c1 = pd.DataFrame(data=c1, index=c.index, columns=c.index)
        
        return c1

    def _ordered_eig(self, x):
        """ Find Real ordered eigenvalues & vectors correlation matrix """
        w, v = LA.eig(np.array(x))    # convert to array & find eig values/vectors
        w = np.real(w)                # convert complex numbers to real
        idx = w.argsort()[::-1]       # eigenvalues aren't ordered automatically
        return w[idx].reshape(w.shape[0], 1), v[:,idx]
    
    
    # %% BOOTSTRAP CHARTS
    # Again the base for these charts may be replicated elsewhere, but seeing
    # as we want these for reporting purposes we've storing them as distinct
    
    # Monte Carlo Simulation paths     
    def plot_paths(self, sims, tgt=0, maxpaths=2500,
                        xtitle='Periods', ytitle='Portfolio Value',
                        template='multi_strat',):
        """ Plots Simulation Paths 
        
        Fairly generic MC Sims path, with an optional target return line - also
        option is ability to cap max paths to stop bloating the chart.
        
        INPUTS:
            sims - sims dataframe OR str name of port in self.results
            tgt - (default tgt = 0) for return bogie
            f - annualisation factor; default is None which uses self.f
            maxpaths - cap on paths
            xtitle & ytitle - fairly obvious
            template - (default multi_strat)
            
        DEVELOPMENT:
            - add colourscale based on rank or terminal value
        """
        
        # check sims
        # if sims is a string we assume it's a port name acessable via results
        # othereise assume a sims dataframe has been passed
        if isinstance(sims, str):
            sims = self.results[sims]['sims']
        else:
            sims = sims
        
        ## BASIC ADMIN
        l, n = sims.shape
        sims = sims - 1
        colour_rgb = 'rgba(180,180,180,{})'.format(0.2)   # grey with light opacity
        
        ## Full Paths Chart 
        fig = px.line(title='Stochastic Return Paths; {}-simulations'.format(l),
                      template=template)
        fig.update_layout(showlegend=False)
        
        # Append sims
        for i, v in enumerate(sims.index):
            
            # plot gets mad (& takes a lot of memory) when nsims is large
            # we can constrain paths to a max number with maxpaths
            if i >= maxpaths:
                continue
            
            # iteration line
            fig.add_scatter(x=sims.columns,
                            y=sims.iloc[i, :],
                            name="",
                            line={'color':colour_rgb, 'width': 0.5})
        
        fig.update_layout(
            yaxis={'anchor':'x1', 'title':ytitle, 'hoverformat':'.1%', 'tickformat':'.0%',},
            xaxis={'anchor':'y1', 'title':xtitle, 'hoverformat':'.0f', 'tickformat':'.0f',})
            
        # target return path
        if tgt != 0:
            tgt_rtn = [1] + [(1 + tgt) ** (1 / self.f)] * (n + 1)
            fig.add_scatter(x=sims.columns,
                            y=(np.cumprod(tgt_rtn)-1),
                            line={'color':'teal'})
        
        return fig
    
    def plot_histogram(self, annsims, portrange=False, tgt=0, 
                       periods=[52], nbins=100,  
                       opacity=0.5, f=None,
                       title='Probability Return Distributions',
                       template='multi_strat',
                       **kwargs):
        """ Histogram of Return Distributions with Boxpot Overlay
        
        INPUTS:
            annsims: dataframe of simulations with annualised returns OR
                     str() name of port in self.results
            tgt: (default tgt = 0) for return bogie; will plot vertical line
            portrange:
                False - (default) annsims is single port_sim & we are plotting 
                         hist for 1 of more periods
                True - annsims is a df with multiple portfolio & single period
            periods: [52] (default) but multiple periods available in list
            nbins: number of bins for historgram
            title: obvious
            opacity: 0-1 (default = 0.5) go lower with more histos on plot
            f: annualisation factor; default is None which uses self.f
            template: (default multi_strat)
            **kwargs: will feed directly into px.histogram()
        """
        
        if isinstance(annsims, str):
            annsims = self.results[annsims]['annsims']
        else:
            annsims = annsims
        
        if f is None:
            f = self.f
        
        #subset data
        annsims = annsims.loc[:, periods]
        
        # reshape for plotly express (stacked format)        
        df = annsims.stack().reset_index()
        if portrange:
            # assumes passed multiple portfolio as same time period
            # rather than 1-portfolio at multiple periods along a simulation
            df.columns = ['sim', 'port', 'returns']
            colour='port'
        else:
            # converting period to string stops the box thing having a shit fit
            df.columns = ['sim', 'period', 'returns']
            df['period'] = 'p-' + df['period'].astype(str)
            colour='period'
            
        # Actual Histogram    
        fig = px.histogram(df, x='returns', color=colour,
                           nbins=nbins,
                           marginal="box",
                           histnorm='probability',
                           histfunc='avg',
                           title=title,
                           template=template,
                           opacity=opacity,
                           **kwargs)
        
        # overlay rather than stacked historgram
        fig.update_layout(barmode='overlay')
        
        # Update Axis
        fig.update_layout(yaxis= {'title':'Probability', 'hoverformat':'.1%', 'tickformat':'.0%',},
                          xaxis= {'title':'Annualised Return', 'hoverformat':'.1%', 'tickformat':'.1%',})
            
        # Add Return Target Vertical Line annotation
        if tgt != 0:
            fig.update_layout(shapes=[dict(type='line',
                                           line={'color':'teal', 'dash':'solid'},
                                           yref='paper', y0=0, y1=0.98, xref='x', x0=tgt, x1=tgt)])
            fig.add_annotation(text="Return Target {:.1%}".format(tgt),
                                xref='x', x=tgt, yref='paper', y=1 , ax=0, ay=0)

        return fig
    
    def plot_ridgeline(self, annsims, traces=[52, 104, 156, 208, 260], width=3,
                 side='positive', meanline=True, box=False,
                 title='Ridgeline KDE Distributions', 
                 template='multi_strat', **kwargs):
        """ Ridgeline Plot
        
        Each specified column (via traces) is converted to KDE distribution
        ploted as seperate trace going up the y-axis 
        
        INPUTS:
            annsims: dataframe of simulations with annualised returns OR
                     str() name of port in self.results
            traces: columns from annsims to turn into ridgelines
            width: (default = 3) is the height
            meanline: True(default)|False - pops a vertical mean in ridge
            box: True|False(default) - box-plot within ridge
            side: 'postive'(default)|'negative' - ridges going up or down
            **kwargs: will feed directly into px.histogram()
        
        REFERENCES:
            https://mathisonian.github.io/kde/
            https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
        
        DEVELOPMENT:
            - look at the hoverdata, probs aren't what I'd like
        
        """
        
        # grab from self if a string input provided for annualised simulations
        if isinstance(annsims, str):
            annsims = self.results[annsims]['annsims']
        
        # subset the data, there is a funny here is the trace list is numerical
        # we first try an iloc and then do a loc if the iloc fails
        try:
            annsims = annsims.iloc[:, traces]    # subset data
        except:
            annsims = annsims.loc[:, traces]
            
        # create a blended colours list- here is teal to purple
        from plotly.colors import n_colors
        colors = n_colors('rgb(0, 128, 128)',
                          'rgb(128, 0, 128)',
                          len(traces),
                          colortype='rgb')
        
        # blank plotly express template
        fig = px.scatter(title=title, template=template)    
        
        for i, v in enumerate(annsims):             # add violin plots as traces
            vn = "p-{:.0f}".format(v) if type(v) == int else v
            fig.add_trace(go.Violin(x=annsims.iloc[:,i],
                                    line_color=colors[i],
                                    line_width=1,
                                    name=vn,
                                    spanmode='soft',))
        
        # convert from violins to horizontal kde charts 
        fig.update_traces(orientation='h', 
                          side=side,
                          meanline_visible=meanline,
                          width=width,
                          box_visible=box)
        
        # update layouts
        fig.update_layout(
            yaxis= {'anchor':'x1', 'title':'Simulation', 'hoverformat':':.1%', 'tickformat':':.0%',},
            xaxis= {'anchor':'y1', 'title':'Annualised Return'})
        
        fig.add_annotation(text="Source: STANLIB Multi-Strategy".format(),
                            xref='paper', x=1, yref='paper', y=-0.085 , ax=0, ay=0, align='right')
        
        return fig
    
    
    # Density Heatmap showing simulated returns through time
    def plot_densitymap(self, sims, f=None, 
                              title='Density Heatmap',
                              xtitle='Simulation Period',
                              ytitle='Annualised Return'):
        
        """ Density Heatmap of Simulations through time
        
        WARNING THIS THING CAN BE MASSIVE - NEED TO SORT OUT WHY
        
        x-axis is shows periods of simulation from 1-year (f) to the end
        y-axis is set to either annualised return or excess return
        colourscale is the probability
        
        INPUTS:
            sims - sims dataframe OR str name of port in self.results
            f - annualisation factor; default is None which uses self.f
        
        NB/ THIS IS STILL A WORK IN PROGRESS
            - need to format colourscale to percentage
            - think we could generalise and make more useful
        """
        
        ## INPUTS
        sims = self.results[sims]['sims'] if isinstance(sims, str) else sims
        f = self.f if f is None else f
        
        sims = sims ** (f/sims.columns) - 1    # annualise returns
        
        # set x axis bins from 1-year to end of simulation
        # subset to monthly because the chart size gets quite bit (5mb)
        nbinsx=sims.shape[1] - f
        
        # stack for Plotly Express
        sims = sims.iloc[:,f:].stack().reset_index()
        sims.columns = ['sim', 'period', 'return']
        
        # Heatmal
        fig = px.density_heatmap(sims, x='period', y='return', nbinsx=nbinsx,
                                 histnorm='percent', histfunc='avg',
                                 labels={'period':xtitle,
                                         'return':ytitle,},
                                 title=title, template='multi_strat')
    
        # Good options are 'Tealrose', 'Greys', 'Purples'
        # https://plotly.com/python/builtin-colorscales/
        fig.update_traces(dict(colorscale ='Tealrose',
                               reversescale = False,
                               showscale=True,
                               coloraxis=None),)

        # formatting
        fig.update_layout(yaxis={'hoverformat':'.1%', 'tickformat':'.0%',},
                          xaxis={'hoverformat':':.0f', 'tickformat':':.0f',},)
    
        # Update hoverdata - NB X and Y are HARDCODED
        fig['data'][0]['hovertemplate'] = 'Sim Period=%{x}<br>Annualised Return=%{y}<br>Prob=%{z}<extra></extra>'
            
        return fig



    # Risk Return of Assets & Plotted Efficient Frontier
    def plot_frontier(self, w=None, mu=None, vol=None, cor=None, template='multi_strat'):
        """ Risk Return Scatter Plot with Efficient Frontier of Portfolios """
        
        ## IMPORTS
        w = self.wgts if w is None else w
        mu = self.mu if mu is None else mu
        vol = self.vol if vol is None else vol
        cor = self.cor if cor is None else cor
                
        # Basics
        idx = w.sum(axis=1) != 0                  # assets with portfolio positions
        vcv = np.diag(vol) @ cor @ np.diag(vol)   # covariance matrix
        
        # Risk Return Scatter
        fig = px.scatter(x=vol[idx], y=mu[idx], text=w[idx].index,
                         #hover_data=[w[idx].index],
                         labels={'x': 'Volatility',
                                 'y': 'Expected Return',},
                         title='Risk Return Chart',
                         template=template)
        
        # Update Asset Class Labels
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')),)
        
        # Portfolio Stats Dataframe
        port_rtn = w.multiply(mu, axis=0).sum().values
        port_vol = [np.sqrt(w[i].values @ vcv @ w[i].values.T) for i in w]
        df = pd.DataFrame([port_rtn, port_vol], columns=w.columns, index=['port_rtn', 'port_vol']).T
        
        # Add Efficient Frontier
        fig.add_scatter(x=df.loc[:,'port_vol'], y=df.loc[:,'port_rtn'],
                        hovertext=df.index,
                        marker=dict(size=10, line=dict(width=0.5), symbol='diamond'),
                        name='Portfolios')
        
        return fig
    
    # Stacked bar chart for portfolio weights
    def plot_wgts_bar_stacked(self, wgts=None, 
                              title='Frontier Portfolio Weights',
                              template='multi_strat'):
        """ Stacked Bar Chart with Portfolio Weights of Multiple Portfolios """
        
        if wgts is None:
            wgts = self.wgts
        
        # Stack end on end for plotly & rename columns (for colours)
        df = wgts[wgts.sum(axis=1) != 0].stack().reset_index()
        df.columns = ['Asset', 'port', 'w']
    
        # plotly bar chart with Plotly Express
        fig = px.bar(df, x='port', y='w', color='Asset',
                     title=title,
                     labels={'port':'Portfolio', 'w': "Port Weight",},
                     template=template,)
        
        return fig    
    
    # Correlation Heatmap
    def plot_correl(self, cor=None, title='Correlation Matrix', aspect='auto',
                    colorscale='Tealrose', reversescale=False, **kwargs):
        """ Plotly Heatmap with Overlay annotations """
        
        # Pull correlation matrix from Bootsrap class
        if cor is None:
            cor = self.cor
        
        ## Basic plotly express imshow heatmap
        fig = px.imshow(cor,
                        x=cor.index, y=cor.index,
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
    
# %% TESTING

def unit_test():
    """ NOT A TRUE UNIT TEST YET """

    # Connect to Excel Workbook & setup lambda function for importing data
    wb = xlw.Book('Viper.xlsm')
    pullxlw = lambda a, b: wb.sheets[a].range(b).options(
        pd.DataFrame, expand='table').value
    
    # Table of Bootstap Imports - mu, vol, alpha, te
    mc = pullxlw('viper', 'A1')
    mu = mc['ExRtn']
    vol = mc['Vol']
    alpha = mc['ALPHA']
    te = mc['TE']
    
    # Table of Port Weights - which has return targets as last row
    wgts = pullxlw('viper', 'J1').iloc[:-1,:]
    tgts = pullxlw('viper', 'J1').iloc[-1,:]
    
    # Historical Returns for Emperical Analysis
    rtns = pullxlw('STATIC ZAR', 'D5').reset_index()
    cor = rtns.corr()

    # set up bootstrap class
    bs = bootstrap(wgts=wgts, mu=mu, vol=vol, hist=rtns, cor=cor, nsims=100)
    
    sim0 = bs.emperical(w=wgts.iloc[:,0])        # run a sim on 1st port
    sst0 = bs.sim_stats(sims=sim0)               # run sim stats on 1st port   
    bs.port_stats()                              # run port stats and consume
    bs.emperical_frontier(alpha=False, tgts=0)   # run emperical frontier
    
    # Chart Tests
    
    #bs.plot_paths('MS4_v1').show()
    #bs.plot_wgts_bar_stacked().show()
    #bs.plot_correl().show()
    #bs.plot_frontier().show()
    #bs.plot_densitymap('MS4_v1').show()
    bs.plot_ridgeline('MS4_v1').show()
    bs.plot_histogram('MS4_v1').show()
    
    return bs

bs = unit_test()
