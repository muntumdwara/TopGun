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

# %% CLASS MODULE

class Bootstrap(object):
    """ Portfolio Stochastic Modelling Class Modules
    
    Currently offers emperical ONLY stochastic modelling for individual ports
    as well as across a range of ports (called frontiers) as well as a range
    of charts, analysis and markdown-report outputs (you need to go run the 
    markdown elsewhere).
    
    INPUTS:
        wgts - dataframe where indices are asset classes & cols are portfolios
        mu - vector of expected returns (as pd.Series) 
        vol - vector of expected volatilies (as pd.Series)
        hist - dataframe of historic returns (NOT index px/levels)
        cor - dataframe of correlation matrix
        nsims - number of Monte Carlo simulations
        psims - no of periods to run simulation over (default = 260w)
        f - annualisation factor (default = 52 for weekly returns data)
    
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
        plot_collection_all(): runs default plots for frontier & ports
        plot_collection_frontier(): runs plots to analyse across portfolios
        plot_collection_port(): runs plots to analyse timeseries of simulations
        
        NB/ for details of individual charts go look below, or run collection
        then load each plotly figures from the collection to see what it is
    
    DEVELOPMENT:
        - check correlation matrix PSD in class properties

    Author: David J McNay
    """

    ## Initialise class
    def __init__(self, wgts, mu, vol,                # these aren't optional 
                 alpha=None, te=None, tgts=None,     # optional
                 hist=None, cor=None,                # Need something
                 nsims=1000, f=52, psims=260,        # standard params
                 **kwargs):
        
        ### ORDER OF INITIALISATION IS IMPORTANT ###

        ### Non-optional class inputs
        self.wgts = wgts
        self.mu = mu          # [has @property]
        self.vol = vol        # [has @property]

        # From required inputs we set these
        self.universe = mu.index          # list of asset classes [has @property]
        self.port_names = wgts.columns    # useful to have names of portfolios

        ### Optional class inputs
        
        # alpha - set to vector of zeros of None passed [has @property]
        if alpha is None:
            alpha = pd.Series(np.zeros(len(mu)), index=mu.index, name='alpha')
        self.alpha = alpha

        # tracking error - set to vector of zeros if None passed [has @property]
        if te is None:
            te = pd.Series(np.zeros(len(mu)), index=mu.index, name='te')
        self.te = te
        
        # tgts set to vector of of zeros of length the numper of portfolios
        if tgts is None:
            tgts = pd.Series(np.zeros(len(wgts.columns)),
                             index=wgts.columns,
                             name='tgts')
        self.tgts = tgts

        # Historical Timeseries Data & Correlation
        # ORDER IMPORTANT HERE
        # if hist provided set a default correlation matrix as RMT
        # if cor also provided we then override the default
        # this is a little inefficient, but meh... hardly matters
        if hist is not None:
            self.cor = self.correl_rmt_filtered(hist.corr())
            self.hist = hist
        
        # Override default correl (from hist) if cor specifically passed
        if cor is not None:
            self.cor = cor        # check symmetrical in properties
        
        ### STANDARD SETTINGS
        self.nsims = nsims    # number of simulations
        self.f = f            # annualisation factor
        self.psims = psims    # no of periods in MC simulation
        self.plots = dict()   # initisalise dict for plotly plots (useful later)

        ## Update Plotly template
        colourmap = ['grey', 'teal', 'purple', 'black', 'deeppink', 'skyblue', 'lime', 'green','darkorange', 'gold', 'navy', 'darkred',]
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
                                    'yref': 'paper', 'y': 0.05, 'ay': 0,}],))
        
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
        
    # Alpha (alpha) - Ideally pd.Series but MUST be a (1xN) vector 
    @property
    def alpha(self): return self.__alpha
    @alpha.getter
    def alpha(self): return self.__alpha
    @alpha.setter
    def alpha(self, x):
        if isinstance(x, pd.Series):
            x.name = 'alpha'
        elif len(np.shape(x)) != 1:
            raise ValueError('alpha input is non-vector: {} given'.format(x))  
        self.__alpha = x
        
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
        
    # Tracking Error (te) - Ideally pd.Series but MUST be a (1xN) vector 
    @property
    def te(self): return self.__te
    @te.getter
    def te(self): return self.__te
    @te.setter
    def te(self, x):
        if isinstance(x, pd.Series):
            x.name = 'te'
        elif len(np.shape(x)) != 1:
            raise ValueError('te input is non-vector: {} given'.format(x))  
        self.__te = x
        
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
        
        # round on the output to save space in chart memory later
        return pd.concat([v1, df], axis=1).round(5)
    
    
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
    
    def plot_histogram(self, annsims=None, portrange=False, tgt=0, 
                       periods=[52], nbins=100,  
                       opacity=0.5, 
                       title='Probability Return Distributions',
                       template='multi_strat',
                       **kwargs):
        """ Histogram of Return Distributions with Boxpot Overlay
        
        INPUTS:
            annsims: None implies entire frontier defined by self.port_names
                     dataframe of simulations with annualised returns OR
                     str() name of port in self.results
            tgt: (default tgt = 0) for return bogie; will plot vertical line
            portrange:
                False - (default) annsims is single port_sim & we are plotting 
                         hist for 1 of more periods
                True - annsims is a df with multiple portfolio & single period
            periods: [52] (default) but multiple periods available in list
                     if going across frontier will set to MAX value in periods
            nbins: number of bins for historgram
            title: obvious
            opacity: 0-1 (default = 0.5) go lower with more histos on plot
            template: (default multi_strat)
            **kwargs: will feed directly into px.histogram()
        """

        # annsims can be the actual sims to be plotted on histrogram or
        # a string with the name of a port in self.results or
        # None in which case we assume a single period, but the whole frontier
        # frontier is defined by self.port_names which is normally the same
        # as self.results.keys() but allows a subset if we have lots of ports
        # and only want a smaller number of those results on the frontier
        if annsims is None:
            
            # if going across frontier we can't have more than 1 period
            # then we iterate through the frontier
            # grab the period vector from the ith port & concat to a dataframe
            periods = [max(periods)] if len(periods) > 0 else periods
            for i, k in enumerate(self.port_names):
                
                x = self.results[k]['annsims'].iloc[:, periods]
                if i == 0:
                    df = pd.DataFrame(x)
                else:
                    df = pd.concat([df, x], axis=1) 
            
            df.columns = self.port_names
            annsims = df    # set annsims as the dataframe of sims now
            
            # using frontier also means portrange must be true
            portrange = True
            
        elif isinstance(annsims, str):
            # If input was a str assume its a portfolio from self.results
            annsims = self.results[annsims]['annsims'].iloc[:, periods]
        else:     
            annsims = annsims.loc[:, periods]    #subset data
        
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
    
    
    def plot_box(self, annsims=None, periods=[52, 156, 260],
                 points=False, boxmean='sd',
                 title='Return Distribution Box Plot',
                 template='multi_strat'):
        
        """ Returns Box PLot
        
        INPUTS:
            annsims: None implies entire frontier defined by self.port_names
                     dataframe of simulations with annualised returns OR
                     str() name of port in self.results
            portrange:
                False - (default) annsims is single port_sim & we are plotting 
                         hist for 1 of more periods
                True - annsims is a df with multiple portfolio & single period
            periods: [52] (default) but multiple periods available in list
                     if going across frontier will set to MAX value in periods
            points: False(default)|'outliers'|True
                    Shows side plot with datum; looks silly with large nsims
            boxmean: 'sd'(default)|True|False
                    Includes dashed mean line in box & diamond for 'sd'
                    Turns notched median off because it looks stupid
            title: obvious
            template: (default multi_strat)
                    
        """
        
        # annsims can be the actual sims to be plotted on histrogram or
        # a string with the name of a port in self.results or
        # None in which case we assume a single period, but the whole frontier
        # frontier is defined by self.port_names which is normally the same
        # as self.results.keys() but allows a subset if we have lots of ports
        # and only want a smaller number of those results on the frontier
        if annsims is None:
            
            # if going across frontier we can't have more than 1 period
            # then we iterate through the frontier
            # grab the period vector from the ith port & concat to a dataframe
            periods = [max(periods)] if len(periods) > 0 else periods
            for i, k in enumerate(self.port_names):
                
                x = self.results[k]['annsims'].iloc[:, periods]
                if i == 0:
                    df = pd.DataFrame(x)
                else:
                    df = pd.concat([df, x], axis=1) 
            
            df.columns = self.port_names
            annsims = df    # set annsims as the dataframe of sims now
            
            # using frontier also means portrange must be true
            portrange = True
            
        elif isinstance(annsims, str):
            # If input was a str assume its a portfolio from self.results
            annsims = self.results[annsims]['annsims'].iloc[:, periods]
        else:     
            annsims = annsims.loc[:, periods]    #subset data
        
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
        fig = px.box(df, x=colour , y='returns',  color=colour, 
                     points=points, notched=True,
                     title=title, 
                     template=template)
        
        if boxmean is not None:
            fig.update_traces(boxmean=boxmean, notched=False)
        
        # Update Axis
        fig.update_layout(yaxis= {'title':'Annualised Return', 'hoverformat':'.1%', 'tickformat':'.0%',},
                          xaxis= {'title':'Portfolio', 'hoverformat':'.1%', 'tickformat':'.1%',})
        
        return fig

    
    def plot_ridgeline(self, annsims=None, traces=[52, 104, 156, 208, 260],
                 side='positive', meanline=True, box=False, width=3,
                 title='Ridgeline KDE Distributions', 
                 template='multi_strat', **kwargs):
        """ Ridgeline Plot
        
        Each specified column (via traces) is converted to KDE distribution
        ploted as seperate trace going up the y-axis 
        
        INPUTS:
            annsims: None implies single period across all self.results
                     dataframe of simulations with annualised returns OR
                     str() name of port in self.results
            traces: columns from annsims to turn into ridgelines
                    for whole frontier we can only have a single period - if
                    traces list len() > 1 we'll pick the MAX len period
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
        
        # number of colours on ridge line - silly point
        ncolours = len(traces)
        
        # annsims can be the actual sims to be plotted on the ridgeline or
        # a string with the name of a port in self.results or
        # None in which case we assume a single period, but the whole frontier
        # frontier is defined by self.port_names which is normally the same
        # as self.results.keys() but allows a subset if we have lots of ports
        # and only want a smaller number of those results on the frontier
        if annsims is None:
            
            # if going across frontier we can't have more than 1 period
            # then we iterate through the frontier
            # grab the period vector from the ith port & concat to a dataframe
            traces = [max(traces)] if len(traces) > 0 else traces
            for i, k in enumerate(self.port_names):
                
                x = self.results[k]['annsims'].iloc[:, traces]
                if i == 0:
                    df = pd.DataFrame(x)
                else:
                    df = pd.concat([df, x], axis=1) 
            
            df.columns = self.port_names
            annsims = df    # set annsims as the dataframe of sims now
            ncolours = len(annsims.columns)    # update ncolours
            
        elif isinstance(annsims, str):
            # grab from self if a string input provided for annualised simulations
            annsims = self.results[annsims]['annsims'].iloc[:, traces]
        
        else:
            # subset the data, there is a funny here is the trace list is numerical
            # we first try an iloc and then do a loc if the iloc fails
            try:
                annsims = annsims.iloc[:, traces]    # iloc for numerical
            except:
                annsims = annsims.loc[:, traces]     # loc for string list
            
        # create a blended colours list- here is teal to purple
        from plotly.colors import n_colors
        colors = n_colors('rgb(0, 128, 128)',
                          'rgb(128, 0, 128)',
                          ncolours,
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

    def plot_convergence(bs, frontier=True, port=None, opacity = 0.2,
                         title='Simulated Confidence Funnel', 
                         template='multi_strat', **kwargs):
        """ Area fill showing confidence intervals over time 
        
        INPUTS:
            frontier: True(default)|False
                when True we use self.port_names & the 25%-75% confidence
                when False we need port to be a string with a portfolio name
            port: str() portfolio name only used if frontier == False
        
        """
        
        
        # In collecting data we need to know if this is a frontier or single port
        # frontier - 25%-75% range for multiple portfolios
        # port - more pairs but only 1 port
        if frontier:
            
            # create 2 dataframes for the upper & lower quartiles of data
            # iterate through results
            for i, port in enumerate(bs.port_names):
                u0 = bs.results[port]['stats'].T.loc[:,'25%']
                l0 = bs.results[port]['stats'].T.loc[:,'75%']
    
                if i == 0:
                    u = pd.DataFrame(u0)
                    l = pd.DataFrame(l0)
                else:
                    u = pd.concat([u, u0], axis=1)
                    l = pd.concat([l, l0], axis=1)
            
            # update column headers (which will say eg. 25% ) to port_names
            u.columns = bs.port_names
            l.columns = bs.port_names
            
            # when we build the chart we iterate to add traces
            # zip to ensure tuples for upper and lower
            pairs = zip(bs.port_names[::-1], bs.port_names[::-1])
            ncolours = len(u.columns)    # number of colours we need
    
        else:
            
            # on a single port this isn't required, but to have a single function we do
            u = bs.results[port]['stats'].T
            l = bs.results[port]['stats'].T
            pairs = [('5%', '95%'), ('10%', '90%'), ('25%', '75%'), ('40%', '60%'), ('50%', '50%')]
            ncolours = len(pairs)
        
        # use plotlys gradient thing to get colours between teal and purple
        # go.Scatter won't let us change opacity directly but we can via rgba
        # plotly colours gives rga NOT rgba - append an opacity alpha to the output
        from plotly.colors import n_colors
    
        colors = n_colors('rgb(0, 128, 128)',
                          'rgba(128, 0, 128)',
                          ncolours,
                          colortype='rgb')
        
        for i, c in enumerate(colors):
            c = c[:3] + 'a' + c[3:]    # convert from rgb to rgba  
            
            # insert opacity alpha into the string
            idx = c.find(')')
            colors[i] = c[:idx] + ", {}".format(opacity) + c[idx:]
        
        ### BUILD THE PLOT
        
        # Set up dummy figure
        fig = px.line(title=title, template='multi_strat', **kwargs)
        fig.update_layout(
            yaxis= {'anchor':'x1','title':'Annualised Return', 'hoverformat':':.1%', 'tickformat':':.1%',},
            xaxis= {'anchor':'y1','title':'Simulation Period', 'hoverformat':':.0f', 'tickformat':':.0f',}, )
    
        for i, v in enumerate(pairs):
    
            # Add upper trace 
            fig.add_trace(go.Scatter(x=l.index, y=l.loc[:,v[0]],
                                     line={'width':0}, fill=None,
                                     showlegend=False,
                                     name="{}".format(str(v[0])),))
    
            fig.add_trace(go.Scatter(x=u.index, y=u.loc[:,v[1]],
                                     line={'width':0},
                                     fill='tonexty',
                                     fillcolor=colors[i],
                                     name="{}".format(str(v[1])),)) 
    
        fig.update_layout(
            yaxis= {'anchor':'x1','title':'Annualised Return', 'hoverformat':'.1%', 'tickformat':'.0%',},
            xaxis= {'anchor':'y1','title':'Simulation Period', 'hoverformat':'.0f', 'tickformat':'.0f',}, )
        
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
                              ytitle='Port Weight',
                              template='multi_strat'):
        """ Stacked Bar Chart with Portfolio Weights of Multiple Portfolios 
        
        Originally designed for weights - but works for any stacked bar.
        We also use this for MCR, TCR, PCR charts by setting wgts = self.pcr
        """
        
        if wgts is None:
            wgts = self.wgts
        
        # Stack end on end for plotly & rename columns (for colours)
        df = wgts[wgts.sum(axis=1) != 0].stack().reset_index()
        df.columns = ['Asset', 'port', 'w']
    
        # plotly bar chart with Plotly Express
        fig = px.bar(df, x='port', y='w', color='Asset',
                     title=title,
                     labels={'port':'Portfolio', 'w':ytitle,},
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
    
    # %% PLOTLY TABLES
    ### I REALLY HATE THESE AND WE NEED A BETTER METHOD OF SHOWING TABLES
    
    def plot_table(self, method='wgts', port=None, title=""):
        """ Table with Headers, Index & Totals Row
        
        INPUT:
            method: None|'risk'|'wgts'(default)
                None will assume port is dataframe and use that
                'risk' takes the inputs from self.results for a given port
                'wgts' takes the inputs from self.wgts
            
            port: should be a string with the name of a port for 'risk'
                  otherwise it's a dataframe if no method provided
        """
    
        # where no method provided assume port is a dataframe    
        if method is None:
            df = port
        
        # useful methods for reporting
        # here we build a table with w, mu, risk etc.. inc. totals
        elif method == 'risk':
            
            # grab modellling inputs table from self.results
            df = self.results[port]['inputs']
            df = df[df.w != 0]
            
            # need to add a totals row
            tot = df.sum(axis=0)
            tot.name = 'TOTAL'
            
            # portfolio expected return & alpha 
            tot.loc['mu'] = df.w.multiply(df.mu, axis=0).sum()
            tot.loc['alpha'] = df.alpha.multiply(df.mu, axis=0).sum()
            
            # set volatility and mcr to zero - the sum is meaningless
            tot.loc['vol', 'mcr'] = 0
            
            df = df.append(tot)    # append to dataframe
        
        elif method == 'wgts':
            df = self.wgts                    # pull weights from class
            df = df[df.sum(axis=1) != 0]    # remove assets with zero wgt
            df = df.append(pd.Series(df.sum(axis=0), name='TOTAL'))
            
        ### MAKE PLOTLY TABLE
        
        # Alternating Row Colours
        # create repeating pattern of desired length
        # make a grey total row as well
        if df.shape[0] % 2 == 0:    # even
            rc = ['white', 'whitesmoke'] * int((df.shape[0]+1)/2)
            rc[-1] = 'grey'
        else:                       # odds
            rc = ['white', 'whitesmoke'] * int(df.shape[0]/2)
            rc = rc + ['grey']
        
        # Form table 
        fig = go.Figure(data=[go.Table(
            columnwidth = [100, 50],
            header=dict(values=list(df.reset_index().columns),
                        fill_color='black',
                        line_color='darkslategray',
                        font={'color':'white', 'size':11},
                        align=['left', 'center']),
             cells=dict(values=df.reset_index().T,
                        format=[[], ['.2%']],    # column text formatting
                        fill_color = ['teal', rc,],
                        line_color='darkslategray',
                        align=['left', 'center'],
                        font={'color':['white', 'black'], 'size':11},))])
        
        # formattingn updates
        fig.update_layout(title=title,
                          height=((df.shape[0] + 1) * 30),     # change height
                          margin = {'l':50, 'r':50, 'b':5, 't':50})   
        
        return fig
    
    def plot_stats_table(self, port, periods=[52, 156, 260], title=''):
        """ Simulation Descriptive Stats Table
        
        Plotly table with simulation period in the index and stats of columns
        
        REFERENCES:
            https://plotly.com/python/table/
            
        AUTHOR: David - but don't ask him to fix it
        """
        
        stats = self.results[port]['stats']
        stats = stats.loc[:, periods]
        
        # Index & Formatting for Stats Table
        z = [('mean', '.1%'), ('median', '.1%'),
             ('prob', '.0%'), ('std', '.1%'),
             ('skew', '.2f'), ('kurtosis', '.2f'),
             ('5%', '.1%'), ('10%', '.1%'), ('25%', '.1%'), ('40%', '.1%'), ('50%', '.1%'),
             ('60%', '.1%'),('75%', '.1%'), ('90%', '.1%'), ('95%', '.1%'),]
    
        idx, fmt = zip(*z)             # unzip
        stats = stats.loc[idx, :].T    # susbset stats table & flip horizontal
            
        # plotly table
        fig = go.Figure(data=[go.Table(
            columnwidth = [150, 60],
            header=dict(values=['Sim Period'] + list(stats.columns),
                        fill_color='black',
                        line_color='darkslategray',
                        font={'color':'white', 'size':11},
                        align=['left', 'center']),
            cells=dict(values=stats.reset_index().T,
                       format=[[], [list(fmt)]],    # column text formatting
                       fill_color = ['teal', ['white','whitesmoke']*50],
                       line_color='darkslategray',
                       align=['left', 'center'],
                       font={'color':['white', 'black'], 'size':11},))])
    
        fig.update_layout(title=title,
                          width=((stats.shape[1] + 1) * 60), 
                          height=((stats.shape[0] + 1) * 40),
                          margin = {'l':50, 'r':50, 'b':5, 't':50})    # change width
        
        return fig    
    
    # %% Plot Collections - saves lots of plots to self.plots()
    
    def plot_collection_all(self):
        """ Run port_collection functions for frontier & all portfolios
        
        In each case plots will be returned to self.plots which is a dictionary
        frontier will be at self.plots['frontier'] while the rest will have the
        port_name as the key.
        
        look at guide for plot_collection_frontier() & plot_collection_port()
        for details of exactly which plots are run... this function is just
        for the default settings.        
        """
        
        # python seems to prefer grabing ones self, manipulating & stuffing back 
        # create dict of plots from self
        plots = self.plots
        
        # iterate through all plots in the self.results dictionary
        for port in self.results.keys():
            p = self.plot_collection_port(port=port, plotly2html=True, digest=True)
            plots[port] = p 
        
        
        # Run frontier & digest
        f = self.plot_collection_frontier(plotly2html=True, digest=True)
        plots['frontier'] = f 
        
        # ingest plots back to self
        self.plots = plots
        return "Mega plot run smashed - look in self.plots"
    
    
    def plot_collection_frontier(self, showplots=False,
                                 plotly2html=True, plotlyjs='cdn',
                                 digest=True):
        """ Create Dictionary of Frontier Plots for use in Reporting
        
        NB/ This includes tables (which require some hacks) remember to remove
        the plotly table if we find a better way of presenting tabular data
        
        REFERENCES:
            https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html-python
            https://plotly.com/python-api-reference/generated/plotly.io.to_html.html
        """
        
        plots=dict()    # create empty dictionary
    
        plots['frontier']= self.plot_frontier()          
        plots['wgts']= self.plot_table(method='wgts')      # table of frontier wgts
        plots['wgts_bar']= self.plot_wgts_bar_stacked()    # stacked wgts bar chart
        plots['pcr']= self.plot_wgts_bar_stacked(
            wgts=self.pcr, ytitle='Contribution-to-Risk',
            title='Asset Class Percentage Contribution to Risk')    # stacked PCR
        #plots['tcr']= self.plot_wgts_bar_stacked(wgts=self.tcr, ytitle='Contribution to Risk', title='Asset Class Contribution to Total Risk')         # stacked PCR
        plots['correl']= self.plot_correl()                # correlation matrix
        
        # iterate adding stats tables for each portfolio
        for p in self.port_names:
            plots['stats_' + p] = self.plot_stats_table(
                                        port=p,
                                        periods=[52, 156, 260],
                                        title="Simulation Stats: {}".format(p))
        
        plots['ridgeline'] = self.plot_ridgeline()    # ridgeline frontier
        plots['hist'] = self.plot_histogram()         # TV histogram of frontier
        plots['box'] = self.plot_box()                # TV boxplot
        
        # convergence plot of inter-quartile range through time for each port
        plots['convergence'] = self.plot_convergence(frontier=True)
        
        # useful in Jupyter Notebooks - just an option to show the plots
        if showplots:
            for p in plots:
                p.show()
        
        # option to convert to html
        # very useful as it allows us to strip the javascript from plotly plots
        # see reference for more info
        for k, v in plots.items():
            plots[k] = v.to_html(full_html=False,
                                 include_plotlyjs=plotlyjs,
                                 default_height=550,
                                 default_width=1000)
            
        # Multiple keys is a bit of a pain in markdown later
        # Create a single 'stats' plot which merges the plotly stats tables
        # look for string 'stats_' in keys & append to dummy list
        # then convert to long html str with double line break between each
        stats = []
        for k, v in plots.items():
            if k[:6] == 'stats_':
                stats.append(v)
                
        stats = '\n \n'.join(stats)    # make long str with line-breaks
        plots['stats'] = stats
            
        # save to self.plots() dictionary by default
        if digest:
            self.plots['frontier'] = plots
        
        return plots
        
    def plot_collection_port(self, port,
                            showplots=False,
                            plotly2html=True, plotlyjs='cdn',
                            digest=True):
        """ Create Dictionary of Single Portfolio Plots for use in Reporting 
        
        REFERENCES:
            https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html-python
            https://plotly.com/python-api-reference/generated/plotly.io.to_html.html
        """
        
        plots=dict()
        
        # Port Risk Table with MCR, TCR etc..
        plots['risk_table'] = self.plot_table(method='risk', port=port, title="{}".format(port))
        
        # Simulation paths - these make file sizes 
        plots['paths'] = self.plot_paths(port)
        plots['stats'] = self.plot_stats_table(
            port=port,
            periods=[52, 156, 260],
            title="Simulation Stats: {}".format(port))
        
        # single period histogram
        plots['hist'] = self.plot_histogram(port, periods=[260])
        plots['ridgeline'] = self.plot_ridgeline(port)
        
        # convergence shows port and 5%, 10%, 25%, 40% & 50% bands
        plots['convergence'] = self.plot_convergence(frontier=False, port=port)
        
        # multi-period histogram
        plots['hist_multi'] = self.plot_histogram(port, periods=[52, 156, 260])
        
        # useful in Jupyter Notebooks - just an option to show the plots
        if showplots:
            for p in plots:
                p.show()
        
        # option to convert to html
        # very useful as it allows us to strip the javascript from plotly plots
        # see reference for more info    
        for k, v in plots.items():
            plots[k] = v.to_html(full_html=False,
                                 include_plotlyjs=plotlyjs,
                                 default_height=550,
                                 default_width=1000)
    
        # save to self.plots() dictionary by default
        if digest:
            self.plots['frontier'] = plots
    
        return plots 
    
    # %% Bootstrap Reporting
    # VERY MUCH WORK IN PROGRESS
    
    def markdown_frontier_report(self, plots=None, title='TEST'):
        """ Markdown report created by appending lines """
        
        # grab oneself
        if plots is None:
            plots = self.plots['frontier']
    
        md = []     # dummy list container - convert to strings later
        
        md.append("## STANLIB Multi-Strategy Stochastic Modelling")
        md.append("### Frontier Report: {}".format(title))
        md.append("For this analysis we generate {nsims} simulated paths \
                  modelling prospective {psims}-week return distributions. \
                  We use an adjusted emperical copula in order to maintain \
                  higher-moments in the distributions and scale standardised \
                  histrocial returns by forward looking estimates of returns \
                  and volatility; historical sample size used is {weeks}-weeks \
                  with factor modelling used to extend some assets \
                      ".format(nsims=self.nsims,
                               psims=self.psims,
                               weeks=self.hist.shape[0]))
        
        md.append("### Portfolio Weights & Ex-Ante Risk & Return Information")
        md.append("{}".format(plots['frontier']))
        md.append("{}".format(plots['wgts']))
        md.append("{}".format(plots['wgts_bar']))
        #md.append("{}".format(plots['tcr']))
        md.append("{}".format(plots['pcr']))
        md.append("{}".format(plots['correl']))
        
        md.append("### Bootstrapped Simulations")
        md.append("{}".format(plots['stats']))
        md.append("Note: Std within descriptive statistics refers to the \
                  standard deviation of the simulated returns at period-X \
                  which is not the expected volatility of the portfolio.")
        
        md.append("{}".format(plots['ridgeline']))
        md.append("{}".format(plots['hist']))
        md.append("{}".format(plots['box']))
        md.append("{}".format(plots['convergence']))
        md.append("Note: Funnel here show the inter-quartile range vis-a-vis time.")
        
        return "\n \n".join(md)    # NEEDS double line-break to render plots
    
    def markdown_port_report(self, port):
        """ Markdown report created by appending lines """
        
        # grab plots (requires self.plot_collection_port() to be have run)
        plots = self.plots[port]
        
        # dummy list container - convert to strings later
        md = []
        
        # Append markdown
        md.append("# STANLIB Multi-Strategy Bootstrap Report")
        md.append("## Simulated Portfolio: {}".format(port))
        md.append("{}".format(plots['risk_table']))
        md.append("{}".format(plots['paths']))
        md.append("{}".format(plots['stats']))
        md.append("{}".format(plots['hist']))
        md.append("{}".format(plots['ridgeline']))
        md.append("{}".format(plots['convergence']))
        md.append("{}".format(plots['hist_multi']))
        
        return "\n \n".join(md)    # NEEDS double line-break to render plots
    
    def report_writer(self, title="TEST_REPORT", md="# TEST", path=""):
        """ Template Report Writer
        
        Takes a markdown document and title & converts to static HTML file
        
        NB/ if markdown isn't rendering as expected look at the spaces in the
        markdown input file. Few known issues:
            1. # HEADER must be the first char on line or it craps out
            2. need at least 1 line break between plots
            3. '  ' double space is new line (which is obviously invisible);
               can use \n but sometimes doesn't work 
        
        DEVELOPMENT:
            Really could do with some CSS or something to style the page
        
        REFERENCES:
            https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf
            http://zetcode.com/python/jinja/
            
        Author: A very exhausted David McNay started Sept '20
        """    
        
        from jinja2 import Template
        from markdown import markdown
        from datetime import date
        
        # this is the boiler plate HTML used by jinja2 in the report
        # Note {{ report_title }} and {{ report_markdown }}
        base_template = """
            <!DOCTYPE html><html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="X-UA-Compatible" content="ie=edge">
                <title>{{ report_title }}</title>
            </head>
            <body>
                {{ report_markdown }}
                <br><br>
                {{ disclaimer }}
            </body></html>"""
        
        disclaimer = "Report generated on {}".format(date.today().strftime("%d %B %Y"))
        
        # set up base template in jinja2 then render report to HTML
        # zetcode blog was quite usful for an HTML rookie to learn
        template = Template(base_template)
        report = template.render(report_title=title,
                                 report_markdown=markdown(md),
                                 disclaimer=markdown(disclaimer))
        
        # full name and filepath to save report
        full_path = path + title + '.html'
        
        # open text file, clear current data & append report
        with open(full_path, 'w+') as f:
            f.truncate(0)      # clear any data currently in file
            f.write(report)    # write report to file
            f.close()          # close file
            
        return "REPORT WRITTEN: {}".format(title)

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

    # set up bootstrap class
    bs = bootstrap(wgts=wgts, mu=mu, vol=vol, hist=rtns, 
                   #alpha=alpha, te=te,
                   nsims=100)
    
    sim0 = bs.emperical(w=wgts.iloc[:,0])        # run a sim on 1st port
    sst0 = bs.sim_stats(sims=sim0)               # run sim stats on 1st port   
    bs.port_stats()                              # run port stats and consume
    bs.emperical_frontier(alpha=False, tgts=0)   # run emperical frontier
    
    # Chart Tests
    
    #from plotly.offline import plot
    pio.renderers.default='browser'
    
    #bs.plot_paths('MS4_v1').show()
    #bs.plot_wgts_bar_stacked().show()
    #bs.plot_correl().show()
    #bs.plot_frontier().show()
    #bs.plot_densitymap('MS4_v1').show()
    #bs.plot_ridgeline().show()
    #bs.plot_ridgeline('MS4).show()
    #bs.plot_histogram().show()
    #bs.plot_histogram('MS4').show()
    #bs.plot_convergence(frontier=False, port='MS4').show()
    #bs.plot_convergence(frontier=True).show()
    #bs.plot_table(method='risk', port='MS4').show()
    #bs.plot_table(method='wgts').show()
    #bs.plot_stats_table(port='MS4').show()
    
    bs.plot_collection_all()
    #md = bs.markdown_frontier_report()
    #bs.report_writer(md=md)
    
    return bs

#bs = unit_test()

# %%

def bootstrap_unit_test():
    """ Not proper unit testing
    
    Create:
        range of dummy 3-asset portfolios (named RP1-RP4)
        vectors of expected returns, volatility, alpha & TE
        pseudo-random 20-year weekly returns with means & std from mu/vol
    
    Set up a Bootstrap class and initialise with dummy data then run all 
    the main functions and test output/plots.
    
    Will annotate with guidance on what answers ought to look like but haven't
    bothered actually providing output figures.
    
    """
    
    ### Setup a Range of 4 Dummy Portfolios (RP1-4) & Dummy Returns
    # Returns are a random normal distribution with 20-years of weekly data
    
    ## Returns & Vols
    # Bootstrap is designed to take pd.Series() as main vector inputs
    universe = ['EQUITY', 'CREDIT', 'RATES']
    mu= pd.Series(data=[0.1, 0.05, 0.01], index=universe, name='ExRtn')
    vol= pd.Series(data=[0.15, 0.08, 0.01], index=universe, name='Std')
    alpha= pd.Series(data=[0.02, 0.01, 0.01], index=universe, name='active')
    te = pd.Series(data=[0.03, 0.03, 0.01], index=universe, name='tracking')
    
    # Dummy Weights Array
    wgts = pd.DataFrame(data=np.array([[0.2, 0.6, 0.2], [0.4, 0.5, 0.1],
                                       [0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]).T,
                        index=universe,
                        columns=['RP1', 'RP2', 'RP3', 'RP4'])
    
    # Create rtns using random gaussian... obviously correl will be bollocks
    rtns = pd.DataFrame(columns=universe)
    for i, v in enumerate(rtns):
        rtns[v] = np.random.normal(mu[i]/52,               # expected return
                                   vol[i] / np.sqrt(52),   # deannualise vol
                                   (20*52))                # 10-years rtns
    
    ### Setup bootsrap class with theoretical 3-asset class portfolio
    bs = Bootstrap(wgts=wgts, mu=mu, vol=vol, hist=rtns,
                  alpha=alpha, te=te, nsims=100, f=52, psims=260,)
    
    # run emperical bootstrap
    bs.emperical_frontier()
    
    ### Now we test all the things - remember to test both frontier & port
    # by making sure the charts work we cover
    #   self.emperical()
    #   self.port_stats()
    #   self.sim_stats()
    
    # render plotly plots to new tab on browser
    pio.renderers.default='browser'
    
    # This will run plot_collection_frontier() & plot_collection_port()
    # Therefore a good test if all the plotting functions are working
    #bs.plot_collection_all()
    
    return bs

#bs = bootstrap_unit_test()