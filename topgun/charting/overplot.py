# -*- coding: utf-8 -*-
""" Overplot

Series of functions designed to help with charting in Plotly

"""

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# %% PLOTLY EXPRESS STANLIB TEMPLATE

# Approximatation of STANLIB colour theme
COLOUR_MAP = {0:'purple',
              1:'turquoise',
              2:'grey',
              3:'black',
              4:'green',
              5:'blue',
              6:'crimson',
              7:'orange',
              8:'mediumvioletred'}

# Hack together basic template
fig = go.Figure(layout=dict(
                      font={'family':'Courier New', 'size':12},
                      plot_bgcolor= 'white',
                      colorway=['grey','turquoise', 'purple', 'lime', 'blue', 'black', 'brown', 'red', 'orange'],
                      showlegend=False,
                      legend={'orientation':'v'},
                      margin = {'l':75, 'r':50, 'b':50, 't':50},
                      xaxis= {'anchor': 'y1', 'title': '',
                              'showline':True, 'linecolor': 'gray',
                              'zeroline':True, 'zerolinewidth':1 , 'zerolinecolor':'whitesmoke',
                              'showgrid': True, 'gridcolor': 'whitesmoke',
                              },
                      yaxis= {'anchor': 'x1', 'title': '', 'hoverformat':'.1f', 'tickformat':'.1f',
                              'showline':True, 'linecolor':'gray',
                              'zeroline':True, 'zerolinewidth':1 , 'zerolinecolor':'whitesmoke',
                              'showgrid': True, 'gridcolor': 'whitesmoke'},
                      updatemenus= [dict(type='buttons',
                                         active=-1, showactive = True,
                                         direction='down',
                                         y=0.5, x=1.1,
                                         pad = {'l':0, 'r':0, 't':0, 'b':0},
                                         buttons=[])],
                      annotations=[],))

# save it
templated_fig = pio.to_templated(fig)
pio.templates['multi_strat'] = templated_fig.layout.template

# %%

def line_stacker(df, template='multi_strat',
                 yaxis_title= '', source='', source_posn=[0.85, 0.08],
                 **kwargs):
    """ Line plot with columns as lines
    
    Plotly express does a cool thing called 'colours' where it loads multiple
    traces which can be clicked on and off on the chart. It does however require
    a f*cking stupid format where all data is in one long vector with repeated 
    dates AND a column of 'colours'... why they don't just let you use the
    varname and have a logical df is beyond me'
    
    INPUT:
        df - dataframe with dates as index; column headers as legend titles
        kwargs - ONLY use arguments you could normally pass to plotly express
    """
    
    # set up the 3 columns we need
    vn = ['date', 'value', 'index']
    z = pd.DataFrame(columns = vn)
        
    # itserate through df concatinating to silly long vector
    for ticker in df:
        i = df[ticker].reset_index()
        i['value'] = ticker
        i.columns = vn
        z = pd.concat([z, i])
        
    # initial figure
    fig = px.line(z, x="date", y="value", color="index",  template=template, **kwargs)
    
    # updates not in kwargs
    fig.update_layout(yaxis_title=yaxis_title)
    fig.update_layout(showlegend=True)
    
    # Add source as annotation (may need some re-jigging ex-post)
    fig.add_annotation(text=source, xref='paper', yref='paper',
                       align='right', ax=0, ay=0,
                       x=source_posn[0], y=source_posn[1])
    
    return fig

# %%
    
