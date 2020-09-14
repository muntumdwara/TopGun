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
               
                <style>
                
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
                
                </style>
                       
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