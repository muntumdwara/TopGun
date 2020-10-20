# -*- coding: utf-8 -*-
"""
TopGun Backtest Class

@author: David
"""

# %% IMPORTs CELL

# Default Imports
import numpy as np
import pandas as pd

# Plotly for charting
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# %% CLASS MODULE

class Backtest(object):
    """
    
    """

    def __init__(self, portrtns, bmkrtns, 
                 benchmark=None, eom = True, freq=12,
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
        self.benchmark = benchmark
        
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
        
        
        ## DataFrame Styler Default
        
        styles = [dict(selector="th",
                       props=[("font-family", "Garamond"),
                              ('padding', "5px 5px"),
                              ("font-size", "15px"),
                              ("background-color", "black"),
                              ("color", "white"),
                              ("text-align", "center"),
                              ('border', '1px solid black')]),
                  
                  dict(selector="td",
                       props=[("font-family", "Garamond"),
                              ('padding', "5px 5px"),
                              ('min-width','70px'),
                              ("font-size", "14px"),
                              ("text-align", "center"),
                              ('border', '1px solid black')]),   
                  
                  dict(selector="caption",
                       props=[("text-align", "right"),
                              ("caption-side", "bottom"),
                              ("font-size", "85%"),
                              ("color", 'grey')]),] 
        
        self.df_styles = styles
        
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

# %% HELPER FUNCTIONS
    
    def _eom(self, x):
        """ Trivial function to ensure End-of-Month Dates in Pandas """
        x.index = x.index + pd.offsets.MonthEnd(0)
        return x
    
# %% BASIC BACKTESTING
        
    def basic_backtest(self):
       """ 
       
       """
       
       if self.benchmark == None:
           bmk = pd.Series(data=0, index=self.portrtns.index, name='BMK')
       else:
           bmk = self.bmkrtns.loc[:,self.benchmark]
           bmk.name = 'BMK'
        
       # Consolidated dataframe for benchmarks & returns
       # Also set up excess returns at the same time
       # Benchmark always in position 0 in dataframe
       self.rtns = pd.concat([bmk, self.portrtns], axis=1).dropna()
       self.xsrtns = self.rtns.subtract(self.rtns.iloc[:,0], axis='rows')
       
       # cumulative returns
       cr = 1 + self.rtns
       cr.iloc[0,:] = 100
       cr = cr.cumprod()
       
       # ingest cumulative returns & excess returns
       # we leave benchmark in postion 0 with values 0 for alpha
       self.cum_rtn = cr
       self.cum_xs_rtn = cr.subtract(cr.iloc[:,0], axis='rows') + 100
       
       # drawdown analysis
       self.drawdown = self.rtns2drawdown(alpha=False)
       self.xs_drawdown = self.rtns2drawdown(alpha=True)
       self.table_drawdown = self.drawdown_table(alpha=False)
       self.table_xs_drawdown = self.drawdown_table(alpha=True)
       
       # rolling period analysis
       for t in [12, 36]:
           ivol = self.rtns.rolling(window=t).std() * np.sqrt(self.freq)
           irtn = cr.pct_change(t)
           irtn_xs = irtn.subtract(irtn.iloc[:,0], axis='rows')
           self.rolling[t] = dict(vol=ivol, rtn=irtn, xsrtn=irtn_xs)
       
       # Run summary table and ingest
       self.summary = self.backtest_summary()
       
       # Extended Correlation Matrix
       # Use BMK, PORT, PORT_XS_RTNS & the bmkrtns indices to form corr matrix
       rtns_wide = pd.concat([self.rtns, self.xsrtns.iloc[:, 1:]], axis=1)
       rtns_wide.columns = list(self.xsrtns.columns) + list(self.xsrtns.columns + '_XS')[1:]
       rtns_wide = pd.concat([rtns_wide, self.bmkrtns], axis=1).dropna()
       self.rtns_wide = rtns_wide
       self.corr = rtns_wide.corr()
        
       return
    

    def rtns2drawdown(self, alpha=True):
        """ Returns-to-Drawdown Timeseries """
    
        # Need to select a method for drawdown
        # if alpha is True use excess returns, otherwise returns        
        rtns = self.xsrtns if alpha else self.rtns
        
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
    
    def drawdown_table(self, alpha=True, dd_threshold=0):
        
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
        
    def backtest_summary(self):
        """ """
        df = pd.DataFrame()
        
        # Annualised Total Return, Vol & Risk-adjusted-return
        df['TR'] = (self.cum_rtn.iloc[-1,:]/100)**(self.freq/len(self.cum_rtn)) - 1
        df['Vol'] = self.rtns.std() * np.sqrt(self.freq)
        df['RaR'] = df.TR / df.Vol
        
        # Beta, Ex-Post Tracking Error & Information Ratio
        df['Beta'] = self.rtns.cov().iloc[:,0] / self.rtns.iloc[:,0].var()
        df['TE'] = self.xsrtns.std() * np.sqrt(self.freq)
        df['IR'] = (df.TR - df.TR[0]) / df.TE
        
        # Drawdown Analysis
        df['Max_Drawdown'] = self.drawdown.min(axis=0)
        df['Max_XS_DD'] = self.xs_drawdown.min(axis=0)
        df['Hitrate'] = self.xsrtns[self.xsrtns > 0].count() / self.rtns.count()
        df['xs_mean'] = self.xsrtns.mean()
        df['xs_worst'] = self.xsrtns.min()
        df['xs_best'] = self.xsrtns.max()
        
        self.summary = df.T
        return df.T

# %% PLOTLY PLOTS
        
    def plot_index(self, df, title="", benchmark=True,
                   yfmt=['.0f', '.2f'], ytitle='Port', height=0):
        
        """ Basic Line Plot in Backtester"""
        
        fig = px.line(df, title=title, labels={'variable':'Port:'}, template='multi_strat', )
        
        if not benchmark:
            fig.data[0]['visible'] = 'legendonly'    # hide bmk
        
        fig.update_layout(
                yaxis= {'anchor':'x1','title':ytitle, 'tickformat':yfmt[0], 'hoverformat':yfmt[1], },
                xaxis= {'anchor':'y1','title':'', 'hoverformat':'%b-%y', 'tickformat':'%b-%y',},)
        
        if height != 0:
            fig.update_layout(height=height)
        
        return fig
    
    
    def plot_ridgeline(self, df, title='Ridgeline KDE Distributions',
                       side='positive', meanline=True, box=False, width=3,
                       template='multi_strat', **kwargs):
            """ Simplified KDE from bootstrapper """
            
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
            
            return fig

    def plot_histo(self, rtn, title='', opacity=0.5, benchmark=False):
    
        fig = px.histogram(rtn, histnorm='probability', 
                            title=title,
                            opacity=opacity,
                            template='multi_strat')
        
        if benchmark != True:
            fig.data[0]['visible'] = 'legendonly'    # hide bmk from histogram
        
        fig.update_layout(barmode='overlay')
        fig.update_layout(
            yaxis= {'anchor':'x1','title':'Probability', 'tickformat':'.0%', 'hoverformat':'.2%', },
            xaxis= {'anchor':'y1','title':'Excess Return', 'tickformat':'.1%', 'hoverformat':'.2%', },)
        
        return fig

    def plot_regression(self, title='', alpha=True):
        """ CAPM Style Regression Plot
        
        Shows the benchmark on the x-axis & port(s) on the y-axis
        OLS regression line plotted through

        """
        
        # stack either the returns or excess returns
        # rename columns as required
        if alpha:
            y = self.xsrtns.stack().reset_index()
            ytitle='Alpha'
            benchmark=False
        else:
            y = self.rtns.stack().reset_index()
            ytitle='Port Return'
            benchmark=False
        
        y.columns = ['Dates', 'Port', 'Returns']    # rename columns
    
        # Repmat benchmark returns & Match columns
        # This is so we can stack - so we can then concat
        x = pd.concat([self.rtns['BMK']] * len(self.xsrtns.columns), axis=1)
        x.columns = self.xsrtns.columns
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
        
        if not benchmark:
            fig.data[0]['visible'] = 'legendonly'    # hide bmk
        
        return fig

    
    def _plot_hitrate(self, df, title='', binary=True):
        
        # Use crosstab to break pd.Series to pd.DataFrame with months x years
        # Cols will be done alphabetically so we manually reorder dataframe
        plots = pd.crosstab(df.index.year,
                            df.index.strftime("%b"),
                            df.values,
                            aggfunc='sum',
                            rownames=['years'],
                            colnames=['months'])
        
        plots = plots.loc[:,['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        
        # Convert excess returns to hit/miss
        if binary:
            plots = plots.applymap(lambda x: 1 if x >= 0 else x)
            plots = plots.applymap(lambda x: -1 if x <= 0 else x)
        
        # Plot
        fig = px.imshow(plots, x=plots.columns.to_list(), y=plots.index.to_list(),
                        title=title,
                        labels=dict(x='Months', y='Years', color='Hit-or-Miss'),
                        color_continuous_midpoint=0, aspect='auto', template='multi_strat')
        
        # Colourscale stuff
        fig.update_traces(dict(colorscale='Tealrose', reversescale=True, showscale=False, coloraxis=None),)
        
        return fig


    def plot_correl(self, cor=None, title='Correlation Matrix', aspect='auto',
                    colorscale='Tealrose', reversescale=False, **kwargs):
        """ Plotly Heatmap with Overlay annotations
        
        NB/ DIRECT RIP FROM BOOTSTRAP - need to consolidate these in Overplot
        """
            
        # Pull correlation matrix from Bootsrap class
        if cor is None:
            cor = self.cor
            
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

    def plot_hitrates(self,
                      min_count=3,
                      show=False,
                      plotly2html=True, plotlyjs='cdn',
                      plot_height=450, plot_width=850):
        
        """ Combined Function Charts & Annual Table
        
        Charts are the Year x Month, binary, hit-or-miss heatmap and
        Table is the annualised hit rate per year - in a styled dataframe
        
        """
        
        plots = dict()        # dummy dictionary for plotly & dataframe
        df = pd.DataFrame()   # dummy dataframe for table
        
        # iterate through each portfolios alpha
        # could be done as a matrix but complexity isn't worth the speed
        for i, p in enumerate(self.xsrtns):
            
            if i == 0:
                continue
            
            ## PLOTLY
            # Get the Year x Month Hitrate Plot
            plots[p] = self._plot_hitrate(self.xsrtns[p],
                                         title="Hit Rate Heatmap: {}".format(p),
                                         binary=True)
            
            ## TABLE
            # Calc the average annual across backtest things
            # use crosstab again to break down by year and month
            # then we can sum across the months
            ix = self.xsrtns.loc[:, p]
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
                
        # Style annual dataframe
        # This shit is tedious - look at the following links if confused
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
        df = df.reset_index()\
               .style.hide_index()\
               .set_table_styles(self.df_styles)\
               .set_caption('Source: STANLIB Multi-Strategy')\
               .set_table_attributes('style="border-collapse:collapse"')\
               .format(formatter="{:.0f}", subset=pd.IndexSlice[:, df.columns[0]])\
               .format(formatter="{:.1%}", subset=pd.IndexSlice[:, df.columns[0:]])\
               .background_gradient('RdYlGn', vmin=0.2, vmax=0.8)\
               .highlight_null(null_color='white')\
               .set_properties(subset=["years"], **{'font-weight':'bold',
                                                    'color':'white',
                                                    'background-color':'teal', })\
               
        plots['annual'] = df
        
        return plots
    
    
    def plot_master(self, plotly2html=True, plotlyjs='cdn',
                    plot_height=450, plot_width=850):
        
        plots = dict()   # dummy dictionary to hold plots
        
        # Total Return & Excess Return
        plots['tr'] = self.plot_index(self.cum_rtn,
                                      title='Cumulative Returns',
                                      ytitle='Index Level')
        
        plots['xsrtn'] = self.plot_index(self.cum_xs_rtn,
                                      title='Excess Returns',
                                      ytitle='Excess Returns',
                                      benchmark=False)
        
        # Return Distributions
        plots['kde_rtns'] = self.plot_ridgeline(
                                    self.rtns,
                                    title='Ridgeline KDE Distributions: Returns')
        plots['kde_alpha'] = self.plot_ridgeline(
                                    self.xsrtns.iloc[:, 1:],
                                    title='Ridgeline KDE Distributions: Excess Returns')
        
        
        # Regression Charts
        plots['regression_rtn'] = self.plot_regression(
                                            alpha=False,
                                            title='Return Regression: Port Returns')
        plots['regression_alpha'] = self.plot_regression(
                                            alpha=True,
                                            title='Return Regression: Excess Returns')
        plots['histogram'] = self.plot_histo(self.xsrtns,
                                    title='Excess Return Distribution')
        
        
        # Drawdown Charts
        plots['drawdown'] = self.plot_index(self.drawdown,
                           title='Drawdown of Returns',
                           yfmt=['.0%', '.2%'], ytitle='Drawdown',
                           benchmark=True,)
        
        plots['xs_drawdown'] = self.plot_index(self.xs_drawdown,
                           title='Drawdown of Excess Returns',
                           yfmt=['.0%', '.2%'], ytitle='Drawdown',
                           benchmark=False,)
        
        # Rolling Plots
        # Rolling Period Charts
        plots['roll_vol'] = self.plot_index(self.rolling[12]['vol'],
                                            title='Rolling Volatility: 12m',
                                            yfmt=['.0%', '.2%'],
                                            ytitle='Volatility',
                                            height=350)
        
        plots['roll_te'] = self.plot_index(
                                self.xsrtns.rolling(window=12).std() * np.sqrt(self.freq),
                                title='Rolling ex-Post TE: 12m',
                                yfmt=['.0%', '.2%'], ytitle='Volatility',
                                benchmark=False, height=350)

        plots['roll_rar'] = self.plot_index(
                                 self.rolling[12]['xsrtn'] / self.rolling[12]['vol'],
                                 title='Risk Adjusted Return: 12m',
                                 yfmt=['.2f', '.2f'], ytitle='XS Rtn / Vol',
                                 benchmark=False, height=350)

        plots['roll_ir'] = self.plot_index(
                           ((self.cum_rtn.iloc[-1,:]/100)**(12/len(self.cum_rtn))-1) / 
                            (self.xsrtns.rolling(12).std()*np.sqrt(self.freq)),
                           title='Rolling Information Ratio: 12m',
                           yfmt=['.1f', '.2f'], ytitle='IR',
                           benchmark=False, height=350)
        
        # Correlation
        plots['correl_wide'] = self.plot_correl(self.corr)
        
        # Convert to HTML
        if plotly2html:
            for k, v in plots.items():
                plots[k] = v.to_html(full_html=False,
                                     include_plotlyjs=plotlyjs,
                                     default_height=plot_height,
                                     default_width=plot_width,
                                     )   
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

# %% REPORTING

    
    def pretty_panda_summary(self):
        
        # This shit is tedious - look at the following links if confused
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
        # https://pbpython.com/styling-pandas.html
        # https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
        
        df = self.backtest_summary()
        
        # duplicate the index in the dataframe
        # we don't just hide because we want to show & reference the index
        m = df.index
        m.name = 'Metric'
        x = pd.concat([m.to_frame(), df], axis=1).fillna(0)
        
        x = x.style.hide_index()\
             .set_table_styles(self.df_styles)\
             .set_caption('Source: STANLIB Multi-Strategy')\
             .set_table_attributes('style="border-collapse:collapse"')\
             .applymap(lambda x: 'color: white' if x== 0 else 'color: black')
        
        # Generally set to 0.1%; few things as 0.02; zeros have white text
        x = x.format(formatter="{:.1%}", subset=pd.IndexSlice[:, x.columns[1:]])\
             .format(formatter="{:.2f}", subset=pd.IndexSlice[['RaR', 'Beta', 'IR'], x.columns[1:]])\
             
        # set the new "index" column
        x = x.set_properties(subset=["Metric"], **{'text-align':'justify',
                                                   'font-weight':'bold',
                                                   'background-color':'teal',
                                                   'color':'white',
                                                   'min-width':'115px'})\
        
        ## Conditional Format Bits
        # These Include the Benchmark
        y = [['TR', 'RaR', 'Max_Drawdown'], x.columns[1:]]
        x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
        x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)
        
        # These only make sense if there is more than one port being tested
        if len(df.columns) > 2:
            y = [['IR', 'Hitrate'], x.columns[2:]]
            x = x.highlight_max(color='lightseagreen', subset=pd.IndexSlice[y[0], y[1]], axis=1)
            x = x.highlight_min(color='crimson', subset=pd.IndexSlice[y[0], y[1]], axis=1)

        self.summary_styled = x
        return x
    
    def pretty_pandas_drawdown(self, alpha=True):
        
        # standard now - pick from dataframe based on if we want exccess rtns
        if alpha:
            x = self.table_xs_drawdown
        else:
            x = self.table_drawdown
        
        # Sort by drawdown & pick only the last however many
        x = x.sort_values(by='drawdown', ascending=False).tail(10)
        
        # useful for indexing - the formating in pandas can't take NaT
        # so need to find the index of potential end dates that won't have ended
        idxna = ~x['recovery'].isna()
        
        # general stuff
        x = x.reset_index().style.hide_index().set_table_styles(self.df_styles)\
                           .set_caption('Source: STANLIB Multi-Strategy')\
                           .set_table_attributes('style="border-collapse:collapse"')
        
        # specific posh formatting
        x = x.format(dict(start='{:%b-%y}', trough='{:%b-%y}', drawdown='{:.1%}'))\
             .format(formatter="{:%b-%y}", subset=pd.IndexSlice[x.index[idxna], ['end']])\
             .background_gradient('RdYlGn', subset='drawdown')
             
             
        return x
    
    
    def markdown_doc(self, title="TEST"):
        
        md = []     # dummy list container - convert to strings later
    
        # Title
        md.append("# STANLIB Multi-Strategy Backtest")
        md.append("## Report: {} \n \n ".format(title))
        
        self.pretty_panda_summary()
        md.append(self.summary_styled.render())
        
        # Cumulative Returns
        md.append("### Portfolio Returns")
        md.append(self.plots['tr'])
        md.append(self.plots['drawdown'])
        md.append(self.pretty_pandas_drawdown(alpha=False).render())
        md.append(self.plots['kde_rtns'])
        md.append(self.plots['regression_rtn'])
        
        # Excess Return
        md.append("### Excess Returns")
        md.append(self.plots['xsrtn'])
        md.append(self.plots['xs_drawdown'])
        md.append(self.pretty_pandas_drawdown(alpha=True).render())
        md.append(self.plots['kde_alpha'])
        md.append(self.plots['histogram'])
        md.append(self.plots['regression_alpha'])
        
        
        # Rolling
        # Rolling Returns
        md.append("### Rolling Period")
        md.append(self.plots['roll_vol'])
        md.append(self.plots['roll_te'])
        md.append(self.plots['roll_rar'])
        md.append(self.plots['roll_ir'])
        
        
        # Hitrate
        md.append("### Hit Rate Analysis")
        md.append("Here we aren't interested in the quantum of return, \
               simply if alpha was positive or negative for a given month. \
               In the annualised analysis we look at the percentage hit-rate \
               over a year, where we have a minimum of 3-observations; \
               heatmaps we show the month-by-month experience with +1 for \
               positive months and -1 for negative months. \n")
        
        md.append(self.plots['hitrate']['annual'].render())
        
        for p in self.plots['hitrate']:
            if p == 'annual':
                continue
            md.append(self.plots['hitrate'][p])      
        
        # Correlation Analysis
        md.append("### Correlation Review")
        md.append("We present the correlation matrix for the full sample period, \
                   showing both the Portfolio returns and the Alpha stream. \
                   Additionally we include a series of strategic asset classes \
                   relevant for multi-asset portfolios. \n ")
        md.append(self.plots['correl_wide'])
        md.append("\n \n")
        
        return "\n \n".join(md)


# %% TEST CODE
        
# import xlwings as xlw

# wb = xlw.Book('BACKTEST.xlsm')

# # index data from timeseries sheet
# benchmarks = wb.sheets['TIMESERIES'].range('D1').options(pd.DataFrame, expand='table').value.iloc[3:,:]

# # convert string to datetime & ensure EOM
# benchmarks.index = pd.to_datetime(benchmarks.index)
# benchmarks.index = benchmarks.index + pd.offsets.MonthEnd(0)

# E = wb.sheets['Enhanced'].range('A1').options(pd.DataFrame, expand='table').value.iloc[:,1]
# C = wb.sheets['Core'].range('A1').options(pd.DataFrame, expand='table').value.iloc[:,1]

# # Convert to month-end dates
# E.index = E.index + pd.offsets.MonthEnd(0)
# C.index = C.index + pd.offsets.MonthEnd(0)

# # Rename Series
# E.name = 'Enhanced'
# C.name = 'Core'

# rtns = benchmarks.loc[:,'SWIX'].pct_change()
# rtns.name = 'BMK'
# rtns = pd.concat([E, C], axis=1).dropna()
# x = 0.3
# rtns['E30'] = rtns['Enhanced'] * x + rtns['Core'] * (1 - x)

# rtns = rtns.loc[:,['E30', 'Core', 'Enhanced']]

# bt = Backtest(rtns, benchmarks, bmks_as_rtns=False, benchmark='SWIX')
# bt.basic_backtest()
# df = bt.backtest_summary()
# df = bt.drawdown_table()
# bt.plot_master()

# md = bt.markdown_doc()
# from topgun.reporting import Reporting
# Reporting().md2html(md=md, title='test')


#print(df)
#x = bt.rolling