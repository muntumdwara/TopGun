# -*- coding: utf-8 -*-
"""
Reporting Classes

Created on Thu Sep 10 16:49:19 2020

@author: David J McNay
"""

class Reporting(object):
    """ Top Gun Reporting Functions
    
    MAIN FUNCTIONS:
        md2html(): converts a markdown to simple static html for sharing
    
    """

    # initialise class
    def __init__(self):
        
        return

    def md2html(self, title="TEST_REPORT", md="# TEST", path=""):
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
                <style>{{ internal_css }}</style>
                <title>{{ report_title }}</title>
            </head>
            <body>
                {{ report_markdown }}
                <br><br>
                {{ disclaimer }}
            </body></html>"""
        
            internal_css = """
            body {background-color: white;
                          width: 850px;
                          max-width: 95%;
                          margin: auto;
                          font-family: Garamond;
                          }
                    
            h1 {color: teal; margin-left: 20px; margin-right: 20px;}
            h2 {color: darkslategray; margin-left: 20px;}
            h3 {color: teal; margin-left: 20px;}
            p {margin: 20px}
            
            table {margin: 0 auto; align-self: center}
            
            """
            
            if isinstance(md, list):
                md = "\n \n".join(md)
            
            disclaimer = []
            disclaimer.append("## Appendix")
            disclaimer.append("Report generated on {}".format(date.today().strftime("%d %B %Y")))
            disclaimer.append("### Disclaimers  \n \n ")
            disclaimer = "\n \n".join(disclaimer)
            
            # set up base template in jinja2 then render report to HTML
            # zetcode blog was quite usful for an HTML rookie to learn
            template = Template(base_template)
            report = template.render(internal_css=internal_css,
                                     report_title=title,
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
        
        
# %% UNIT TESTING
            
#Reporting().md2html()

# %% REPORTS

#import xlwings as xlw
#import pandas as pd
#from topgun import Bootstrap

# def bootstrap_report_smash(title='TEST', 
#                            active=False,
#                            path="",
#                            n=100,
#                            reports=True):
    
#     """ End-to-End Emperical Stochastic Modelling
    
#     Function:
#         1. Pulls Data Required for MC Simulation from Excel Workbook
#         2. Runs Stochastic Modelling across all portfolios
#         3. Generates styled HTML reports & saves to folder
        
#     INPUTS:
#         XLWINGS CURRENTLY HARDCODED BUT EASY TO EDIT
#     """
    
#     ### Set up excel wings connection
#     wb = xlw.Book('Viper.xlsm')
#     pullxlw = lambda a, b: wb.sheets[a].range(b).options(pd.DataFrame, expand='table').value

#     ## Pull bootstrap data from Excel
#     mc = pullxlw('viper', 'A1')    # Monte-Carlo Input Table & Weights
#     wgts = pullxlw('viper', 'J1')  # pull wgts cols (return tgts at the bottom)
#     rtns = pullxlw('STATIC ZAR', 'D5').reset_index()

#     ## Manipulate
#     mu, vol = mc['ExRtn'], mc['Vol']
#     alpha, te = mc['ALPHA'], mc['TE']

#     # Order matters for these
#     wgts = wgts.iloc[:-1,:]        # now strip return tgt from wgts
    
#     ### setup bootstrap class
#     bs = Bootstrap(wgts=wgts, mu=mu, vol=vol, hist=rtns, nsims=n, psims=260, f=52)
    
#     if active:
#         bs.alpha = alpha
#         bs.te = te
    
#     if reports:
    
#         # run emperical monte-carlo class
#         #display("Bootstrap: Running Simulations")
#         _ = bs.empirical_frontier()

#         #display("Bookstrap: Simulations complete; generating Plots")
#         _ = bs.plot_collection_all()

#         # produce reports
#         #display("Bootstrap: Writing Reports")
#         md = bs.markdown_master(title=title)
#         Reporting().md2html(title=title, md=md, path=path)

    
#     return bs

#bs = bootstrap_report_smash(title="TEST", n=10000, active=True)

























