# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:52:21 2020



@author: David McNay
"""


# %% Packages

# Standard imports
import numpy as np
import pandas as pd
import xlwings as xlw
import math
import re
import datetime

# Charting packages
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# %% Import Data

### DUKE Import & Manipulation
def _import_duke(filename='Duke.xlsx'):
    
    ## Import from xlwings
    duke = xlw.Book(filename).sheets[0].range('A5').options(pd.DataFrame, expand='table').value.reset_index()
    
    ## The Duke dataset has A LOT of hidden white spaces & special characters
    # Remove special characters, the strip left and right trailing spaces
    for i, vn in enumerate(duke.columns.values):
        vn1 = re.sub('[^\w\s]', '', vn).rstrip().lstrip()
        duke.rename(columns={vn:vn1}, inplace=True)    
    
    ## Now subset DataFrame for what we need & rename some columns we use a lot
    duke_col_names = {'Company and Scientific Name':'ScientificName',
                      'Purchaser Entity  Country': 'PurchaserEntity',
                      'Purchasers Country Economic  Status':'EconomicStatus',
                      'Number of Doses Procured': 'DosesProcured',
                      'Number of Doses Needed per Person':'DosesRequired',
                      'Population': 'Population',
                      'Doses intended to be purchased': 'OptionalDoses',}

    duke = duke.iloc[:, duke.columns.isin(duke_col_names.keys())]    # subset
    duke.rename(columns=duke_col_names, inplace=True)                # rename columns
    
    ## Now ensure string data has no trailing whitespaces and numerical data is actually numerical
    idx = duke.columns.isin(['ScientificName', 'PurchaserEntity', 'EconomicStatus'])   
    duke.loc[:, idx] = duke.loc[:, idx].fillna("")
    duke.loc[:, idx] = duke.loc[:, idx].applymap(lambda x: x.lstrip().rstrip())
    
    # Numerical columns
    idx = duke.columns.isin(['DosesProcured', 'OptionalDoses', 'Population', 'OptionalDoses'])
    duke.iloc[:, idx] = duke.iloc[:, idx].astype(float)
    
    # For doses & optional we want blanks to be zeros; for doses required & population make zero
    idx = duke.columns.isin(['DosesProcured', 'OptionalDoses'])
    duke.loc[:, idx] = duke.loc[:, idx].replace(np.nan, 0)
    
    idx = duke.columns.isin(['DosesRequired', 'Population'])
    duke.loc[:, idx] = duke.loc[:, idx].fillna(0)
    
    
    ## Next Re-order Vaccines from Duke list & add a ShortName
    # This is a manually selected list and order (from df dukekey)
    dukekey = xlw.Book('Data.xlsx').sheets['KEY'].range('C2').options(pd.DataFrame, expand='table').value
    duke['ShortName'] = duke['ScientificName'].map(dukekey.to_dict()['ShortName'])    # map shortname
    
    # Some jiggery pokery to adjust the order of the vaccines in the Duke List
    # I specify a bespoke order in the data key No 1-16
    # We sort index on multi-index then drop working columns; little hack with the myorder2
    vaccineorder = dukekey['Order'].to_dict()                     # get bespoke order
    duke['myorder'] = duke['ScientificName'].map(vaccineorder)    # add bespoke order
    duke['myorder2'] = 1 / duke.DosesProcured                     # this is a hack so we can sort descending
    duke = duke.set_index(['myorder', 'myorder2']).sort_index().reset_index().drop(columns=['myorder', 'myorder2'])
    
    ## Find total doses
    # Drop any where the total doses == 0; there are probably contract negotations in place
    duke['DosesTotal'] = duke.DosesProcured + duke.OptionalDoses
    duke = duke[duke.DosesTotal != 0]
    
    ## Basic Analysis
    # No of people vaccinated (doses/required doses)
    duke['VaxProcured'] = duke.DosesProcured / duke.DosesRequired
    duke['VaxOptional'] = duke.OptionalDoses / duke.DosesRequired
    duke['VaxTotal'] = duke.DosesTotal / duke.DosesRequired
    
    # As percentage of Population
    duke['PopVaxProcured'] = duke.VaxProcured / duke.Population
    duke['PopVaxOptional'] = duke.VaxOptional / duke.Population
    duke['PopVaxTotal'] = duke.VaxTotal / duke.Population    
    
    return duke


def _import_other_stuff(filename='Data.xlsx'):
    
    ### Import Non-Duke Data
    # Where relevant ensure all dates are month-end

    ## Vaccine Data
    # This is the table of manually collected data by me
    vx = xlw.Book('Data.xlsx').sheets['VACCINE'].range('A2').options(pd.DataFrame, expand='table').value
    vx = vx.iloc[:, :list(vx.columns).index("Source")]    # drop columns after "Source"
    for c in ['ramp_up', 'approved', 'full_capacity']:
        vx[c] = vx[c] + pd.offsets.MonthEnd(0)
    
    ## Region & Population Data
    # From KEY tab, this combines Country, Economic Group, Population & "UN" group for demographics
    # Don't really need classifications within Low & Middle Income
    regionmap = xlw.Book('Data.xlsx').sheets['KEY'].range('H2').options(pd.DataFrame, expand='table').value
    regionmap['Wealth'] = regionmap['EconomicGroup'].map({'High income':'HI',
                                                          'Upper middle income':'LMI',
                                                          'Lower middle income':'LMI',
                                                          'Global Entity':'Global'})
    
    ## UN Demographics Information
    # UN do this by continent so actually supplemented with census data in some places 
    un = xlw.Book('Data.xlsx').sheets['UN_Data'].range('B2').options(pd.DataFrame, expand='table').value
    
    ## Override Tab
    # THis is where we manually allocated vaccines
    # We remove notes columns because not really used in Python model
    override = xlw.Book('Data.xlsx').sheets['OVERRIDE'].range('A2').options(pd.DataFrame, expand='table').value
    override['DATE'] = override['DATE'] + pd.offsets.QuarterEnd(0)
    override.reset_index(inplace=True)
    override = override.iloc[:,:list(override.columns).index("NOTES")]    # remove notes columns
    
    return vx, regionmap, un, override

# %% Month-on-Month Manufacture

def mom_manufacture(vx, startdate='2020-12-31', order=1):
    """ Month on Month Manufacture Model
    
    * Only uses the vx input from excel
    * Models production based on 'steady', 'step' or 'growth'
        steady - constant mom production [ongoing / 12]
        step - constant through 2021 then new constant in 2022
        growth - linear increase from 'start_cap' to 'ongoing' arriving at full
            capacity on 'ongoing' date. If 'doses_2021' is not 0 then the implied
            mom manufacture will be scalled to be exactly 'doses_2021'; pre-ramp
            up will still be set to zero
    * Set manufacture to zero on months before ramp up
    * Adds 2020 all in one month at the end
    """
    
    idx = pd.date_range(startdate, periods=36+1, freq='m')
    mom = pd.DataFrame(index=idx, columns=vx.index, data=np.nan)
    
    for i, v in enumerate(mom):

        # Steady is simple, we assume MoM manufacturing is constant
        if vx.loc[v, 'mfg_model'] == 'steady':
            mom.loc[:, v] = vx.loc[v, 'ongoing'] / 12

        # Step is a production step up on given day (currently Jan '22)
        elif vx.loc[v, 'mfg_model'] == 'step':

            step_date = '31/12/2021'
            mom.loc[mom.index <= step_date, v] = vx.loc[v, 'doses_2021'] / 12
            mom.loc[mom.index > step_date, v] = vx.loc[v, 'ongoing'] / 12

        # Increase in production as factories come on line
        # Currently linear but could be updated
        elif vx.loc[v, 'mfg_model'] == 'growth':

            # get starting mom production & terminal mom production
            mom.loc[mom.index[0], v] = vx.loc[v, 'start_cap'] / 12
            mom.loc[vx.loc[v, 'full_capacity'], v] = vx.loc[v, 'ongoing'] / 12

            # Interpolation could be done outside IF in 1-step but this is easier to follow
            mom.loc[:, v] = mom.loc[:, v].interpolate(method='polynomial', order=order).ffill()
            
            # Scale 2021 - where we have a 2021 production target
            if vx.loc[v, 'doses_2021'] != 0:
                mom.loc[mom.index <= vx.loc[v, 'ramp_up'], v] = 0
                p0 = mom.loc[mom.index <'01/01/2022', v]
                p1 = p0 * (vx.loc[v, 'doses_2021'] / p0.sum())
                mom.loc[mom.index <'01/01/2022', v] = p1

        # Final step to remove all production before or equal to the "ramp up" phase
        mom.loc[mom.index <= vx.loc[v, 'ramp_up'], v] = 0

    # Add back 2020 supply as a point-smash in December
    mom.iloc[0, :] = vx.loc[:, 'doses_2020']
        
    # You can't have partial vaccines
    return mom.fillna(0).round(0)     


# %%
    
duke = _import_duke()
vx, regionmap, un, override = _import_other_stuff()
mom = mom_manufacture(vx)

# %% Allocation Model v2

def _override_check(v, t, mfg):
    
    a = pd.Series(name="allocation", dtype=float)    # dummy series
    ovr = override[override['VACCINE'] == v]
    
    # Overrides are specified on a quarterly basis
    # convert current t to quater end
    qtr_end = t - pd.tseries.offsets.QuarterEnd(0)
    
    if qtr_end in set(pd.to_datetime(ovr['DATE'])):
                
        # Subset to date & remove [Vaccine, Date, Method] columns
        ot = ovr[ovr['DATE'] == qtr_end].iloc[:, 2:].copy()
                
        # First convert percentages to actual doses of manufacture
        # If Actual smooth over quarter if after 31/12/20
        # Doses started in Dec-20 so those actual are for Dec
        if ot['METHOD'].iloc[0] == 'PERCENT':
            ot = ot.applymap(lambda x: (x * mfg).round(0) if type(x)==float else x)
        elif (ot['METHOD'].iloc[0] == 'ACTUAL') and (qtr_end != '31/12/2020'):     
            ot = ot.applymap(lambda x: x/3 if type(x)==float else x)
               
        # Don't need the Override Method any more
        ot = ot.drop(columns='METHOD').fillna(np.nan).dropna(axis=1).T
        ot.columns = [t]
        
        # iterate over countries
        for c in ot.index:
            if mfg >= ot.loc[c, t]:
                a[c] = ot.loc[c, t]
                mfg -= ot.loc[c, t]
            else:
                a[c] = mfg
                mfg = 0

    return a, mfg.round(0)

def _proratification(x, mfg, split):
    """
    Logic:
        - find remaining doses to be allocated - which will be the adjusted sum
        - remove any lines where the we have already allocated (so where adjusted == 0)
        - recalaculate a pro-rata allocation for those remaining vaccines
        - 
    """
    
    x = x[x.doses != 0]
    a0 = pd.Series(name="allocation", dtype=float)
    
    prorate = lambda y: (x.loc[:, y] / x.loc[:, y].sum())
    x.loc[:, 'doses_prorata'] = prorate('doses')
    x.loc[:, 'pop_prorata'] = prorate('population')
    x.loc[:, 'risk_prorata'] = prorate('risk')
    x.loc[:, 'adjusted'] = (((x.doses_prorata * split[0]) + 
                              (x.pop_prorata * split[1]) + 
                              (x.risk_prorata * split[2])) * mfg).round(0)
    
    #print(mfg - x.adjusted.sum())
    
    for c in x.index:
        
        # Doses Remaining > Allocation
        if x.loc[c, 'doses'] >= x.loc[c, 'adjusted']:
            a0[c] = x.loc[c, 'adjusted']                  # allocate doses
            x.loc[c, 'doses'] = x.loc[c, 'doses'] - x.loc[c, 'adjusted']    # reduce quantity in x
            x.loc[c, 'adjusted'] = 0                     # set adjusted to 0

        # Where country being allocated more doses than they have remaining
        elif x.loc[c, 'doses'] < x.loc[c, 'adjusted']:
            a0[c] = x.loc[c, 'doses']                             # allocate all remaining doses
            x.loc[c, 'adjusted'] = x.loc[c, 'adjusted'] - x.loc[c, 'doses']
            x.loc[c, 'doses'] = 0
            
    return x, a0

    
def _allocate_doses_mfg(a, mfg, x, exclude, split, group=False):
    
    x = x.copy()
    spares = 'AVAILABLE'
    
    # Pro-Rata Key
    # Population Key (how to account for COVAX here)
    x['wealth'] = x['country'].map(regionmap['Wealth'].to_dict())
    x['population'] = x['country'].map(regionmap['Population'].to_dict())
    if 'COVAX' in set(x.country):
        covax_pop = regionmap.loc['USA', 'Population']
        x.loc[x.country.isin(['COVAX']), 'population'] = covax_pop 
    x['risk'] = x['country'].map(regionmap['50+'].to_dict())
    x.risk = x.risk * x.population
    x.set_index('country', inplace=True)    # for ease set country as index
    
    # Remove countries we need to exclude because of overrides
    # Do we need to subset by wealth group i.e Lower & Middle Income (LMI)
    x = x[~x.index.isin(exclude)]
    if group != False:
        x = x[x.wealth == group]
        spares = 'LMI'
    
    # in theory this is 0, but leave room to lose a few doses in rounding
    while mfg > 5:
        
        x, a0 = _proratification(x, mfg, split)
        a = pd.concat([a, a0], axis=1).sum(axis=1).rename('allocation')
        
        # Basically the months after all pre-orders have been fulfilled
        if len(x) == 0:
            a0 = pd.Series({spares:mfg}, name='allocation')
            mfg = 0
            a = pd.concat([a, a0], axis=1).sum(axis=1).rename('allocation')
            return a
        
        # Otherwise normal case
        mfg = x.adjusted.sum()
        
        # if remaining doses are 0 but adjusted > 0; means we have SPARE DOSES
        if x.doses.sum() == 0:
            if group != False:
                a[group] = x.adjusted.sum()
            else:
                a[spares] = x.adjusted.sum()
                mfg = mfg - x.adjusted.sum()
    return a

def allocation_model2(mom, duke, split=[0.5, 0.25, 0.25], waste=0.1, fudge = {'Chinese':0.25, 'Gamaleya':0.25}):
    
    # Create Tally Card to compare outstanding pre-orders & optional doses
    # Will eventually go to zero as all doses are allocated
    idx = duke.columns.isin(['ShortName', 'PurchaserEntity', 'DosesTotal'])
    tally = duke.loc[:, idx].copy()
    tally = tally.rename(columns={'PurchaserEntity':'country', 'DosesTotal':'doses'})
    
    # Update different Chinese Vaccines in Tally to just be generic Chinese
    cn = 'Chinese'
    tally['ShortName'] = tally['ShortName'].replace({'CanSino':cn, 'Sinovac':cn, 'Sinopharm':cn})
    tally = tally.groupby(by=['ShortName', 'country']).sum().reset_index()
    
    allocation = pd.DataFrame()    # dummy FINAL allocation dataframe
    
    # Iterate over each vaccine available in mom manufacture
    # Then iterate over time
    for v in mom.columns:  
        for t in mom.index:
                
            # dummy series for allocations of this vaccine; concat to allocation later
            a = pd.Series(name="allocation", dtype=float)
            mfg = mom.loc[t, v]        # extract no of doses manufactured from  mom
            mfg = mfg * (1 - waste)    # shrink by wastage
            mfg = mfg * fudge[v] if v in fudge.keys() else mfg
            
            # Check for Overrides
            # Then append overrides to a; country names will be easy to filter later
            # Need to do an allocation regime for LMI allocations
            ovr, mfg = _override_check(v, t, mfg)    # look up for details
            a = pd.concat([a, ovr])                  # append overrides
            exclude = list(ovr.index)
            
            # Low & Middle Income Allocation
            if 'LMI' in set(ovr.index):
                exclude = list(ovr.index)
                a = _allocate_doses_mfg(a, ovr.loc['LMI'], 
                                        tally.loc[tally.ShortName == v],
                                        exclude=exclude, split=split,
                                        group=False)
                mfg = mfg - ovr.loc['LMI']
            
            # Main Allocation
            exclude = list(a.index)
            a = _allocate_doses_mfg(a, mfg, tally.loc[tally.ShortName == v],
                                    exclude=exclude, split=split, group=False)
        
            # Update the Tally card (of remining pre-orders)
            # Remember Tally has ALL vaccine not just v, so need to find and index
            # using both the NAME and the Country
            for c, doses in a.items():                
                idx = tally[(tally.ShortName == v) & (tally.country == c)].index                
                tally.loc[idx, 'doses'] = tally.loc[idx, 'doses'] - doses
            
            # Make some adjustments - convert a to a frame & add t and v
            # Then concat to the master allocation output dataframe
            a1 = a.copy().to_frame().reset_index().rename(columns={'index':'country'})
            a1['date'] = t
            a1['vaccine'] = v
            allocation = pd.concat([allocation, a1])
    
    # Tidy up & add courses
    allocation = allocation.reset_index().set_index(['date', 'country', 'vaccine']).reset_index()
    allocation['courses'] = allocation.vaccine.map(vx.doses.to_dict())
    allocation['courses'] = allocation.allocation / allocation.courses
            
    return allocation.drop(columns='index')


allocation = allocation_model2(mom, duke)

# %% Allocation Test

def test_allocation_function(vaccine='Pfizer'):

    alloc = allocation_model2(mom.loc[:, vaccine].to_frame(), duke)
    
    x = alloc.loc[:,['date', 'country', 'allocation']].pivot(index="date", columns="country", values="allocation")
    xlw.Book('Data.xlsx').sheets['xlw_test1'].range('A1').value = x
    
    y = mom.loc[x.index, vaccine]
    xlw.Book('Data.xlsx').sheets['xlw_test1'].range('Z1').value = y
    
    print('complete')
    
    return

# %%

def popvax(allocation, V=1.95, avail2dm=0.5, lag=1):
    
    # Take long allocation frame & make it courses by country
    # allocation df has columns = ['date', 'vaccine', 'allocation', 'courses']
    # Add South Africa if it is still missing in the Duke data
    abc = allocation.groupby(by=['date', 'country']).sum().reset_index()
    abc = abc.pivot(index="date", columns="country", values="courses")
    if 'SA' not in set(abc.columns):
        abc['SA'] = np.nan
    if 'LMI' not in set(abc.columns):
        abc['LMI'] = np.nan
    abc.fillna(0, inplace=True)    # also replace NaN with zeros

    popvax = pd.DataFrame(data=0, index=abc.index, columns=abc.columns)
    
    # Useful Stuff
    hi = regionmap[regionmap.Wealth == 'HI']
    
    
    lmi = regionmap[regionmap.Wealth == 'LMI']
    lmi = lmi[~lmi.index.isin(['China'])]    # remove China
    
    lmi_pop = lmi.loc[:, 'Population'] / regionmap.loc['Total_LMI_ex_China', 'Population']
    lmi_pop_at_risk = regionmap.loc['Total_LMI_ex_China', 'Population'] * regionmap.loc['World', '50+'] 
    lmi_risk = (lmi.loc[:, 'Population'] * lmi.loc[:, '50+']) / lmi_pop_at_risk
    
    key = (lmi_pop * 0.5) + (lmi_risk * 0.5)
    
    for i, t in enumerate(abc.index):
        
        # Iterate through each HI country; checking vs our cntry df
        for c in hi.index:
            if c in abc.columns:

                # Find current vax (cvax) and monthly vax (mvax)
                cvax = 0 if i == 0 else popvax.iloc[i-1, :].loc[c]
                mvax = abc.loc[t, c] / regionmap.loc[c, 'Population']
    
                # Decide what to do with allocation
                #  1. cvax > V allready maxed out - give away allocation
                #  2. cvax + mvax < V use all the allocation
                #  3. cvax + mvax > V use some and give away the rest
                if cvax > V:
                    abc.loc[t, 'AVAILABLE'] = abc.loc[t, 'AVAILABLE'] + abc.loc[t, c]
                    popvax.loc[t, c] = cvax
                elif (cvax + mvax) < V:
                    popvax.loc[t, c] = cvax + mvax
                elif (cvax + mvax) > V:
                    req = math.ceil((V - cvax) * regionmap.loc[c, 'Population'])    # courses required
                    mvax = req / regionmap.loc[c, 'Population']
                    popvax.loc[t, c] = cvax + mvax
                    abc.loc[t, 'AVAILABLE'] = abc.loc[t, 'AVAILABLE'] + abc.loc[t, c] - req
                    
        # Next we have to allocate the AVAILABLE Supply
        # Unlike v1 we don't make an assumption this all goes to COVAX
        hi_avail = abc.loc[t, 'AVAILABLE'] * avail2dm
        lmi_avail = abc.loc[t, 'AVAILABLE'] * (1 - avail2dm)
        
        if hi_avail > 0:
            cap = V
            idx = popvax.columns.intersection(hi.index)
            hi_sub = popvax.loc[t, idx]
            hi_sub = hi_sub[hi_sub < cap]
            if len(hi_sub) < 1:
                lmi_avail = lmi_avail + hi_avail
            else:
                
                hi_key = hi.loc[hi_sub.index, 'Population'] / hi.loc[hi_sub.index, 'Population'].sum()
                hi_alloc = hi_key * hi_avail
                req = (cap - popvax.loc[t, hi_sub.index]) * hi.loc[hi_sub.index, 'Population']
                req = req.apply(math.ceil)
                idx = hi_alloc >= req
                hi_alloc.loc[idx] = req.loc[idx]
                hi_pc = hi_alloc / hi.loc[hi_sub.index, 'Population']
                popvax.loc[t, hi_pc.index] = popvax.loc[t, hi_pc.index] + hi_pc
                lmi_avail = lmi_avail + (hi_avail - hi_alloc.sum()).round(0)
                
        # allocate LMI avail to COVAX
        abc.loc[t, 'COVAX'] = abc.loc[t, 'COVAX'] + lmi_avail
        
        idx = abc.columns.intersection(key.index)    # itersection index between cntry & key
        for charity in ['LMI', 'COVAX']:
            
            # Create series based off key then re-allocate
            c0 = (abc.loc[t, charity] * key).fillna(0)
            abc.loc[t, idx] = abc.loc[t, idx] + c0.loc[idx] 
            abc.loc[t, charity] = 0    # Do we want to remove allocation from COVAX in table?
        
        for c in lmi.index:
            if c in abc.columns:
                cvax = 0 if i == 0 else popvax.iloc[i-1, :].loc[c]
                mvax = abc.loc[t, c] / regionmap.loc[c, 'Population']
                #popvax.loc[t, c] = cvax + mvax

                if cvax > 1.25:
                   popvax.loc[t, c] = cvax
                else:
                   popvax.loc[t, c] = cvax + mvax   

    return popvax.drop(columns=['AVAILABLE', 'COVAX', 'LMI'])

popvax = popvax(allocation = allocation)
xlw.Book('Data.xlsx').sheets['xlw_popvax'].range('A1').value = popvax 

#abc = allocation.loc[:,['date', 'country', 'courses']].pivot(index="date", columns="country", values="courses")




