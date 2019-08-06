#!/usr/bin/env python
# coding: utf-8

import math
import re
import os.path
from pathlib import Path
import math

import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas import DataFrame, Series
from scipy.interpolate import griddata
from scipy.constants import mu_0
from scipy.integrate import simps
import matplotlib.pyplot as plt
import loaddatasets as ld

FIG_FILE_PATH = Path('.')
PLOT_EXTENSION = ".pdf"

FIGSIZE_IN = (8,6)
DPI = 800

# stolen from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24


N_POINTS_LINEPLOT = 2000

DENSITY_GD = 7900 # kg/m3, from Trevizoli et al (2016), IJR volume 72

B_LABEL = r'$\overline{B}_\mathrm{high}\,[\mathrm{T}]$'
Q_LABEL = r'$\dot{Q}_{\mathrm{C}}\,[\mathrm{W}]$'
COP_LABEL = r'$\mathrm{COP}$'
W_PUMP_LABEL = r'$\w_\mathrm{pump}\,[\mathrm{W}]$'
PHI_LABEL = r'$\Phi$'
H_REG_LABEL = r'$H_\mathrm{r}\,[\mathrm{mm}]$'
F_B_LABEL = r'$F_\mathrm{B}\,[\%]$'
RATIO_F_LABEL = r'$\frac{F_\mathrm{M}}{F_\mathrm{B}}$'
F_M_LABEL = r'$F_\mathrm{M}\,[\%]$'
H_MAX_LABEL = r'${B}_\mathrm{max}\,[\mathrm{T}]$'

plt.rc('text', usetex=True)
plt.rc(
    'font', 
    size=MEDIUM_SIZE,
    family='serif',
    serif='Palatino')          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE,figsize=FIGSIZE_IN)  # fontsize of the figure title

def refine_list(original_list, factor):
    """
    Return a new list, inserting more elements between the number in
    'original_list'.
    
    Assumes 'original_list' is evenly-spaced. The spacing between each
    element is divided by 'factor'
    
    >>>refine_list([1,2,3],factor=2)
    [1.0,1.5,2.0,2.5,3.0]
    """
    
    # calculate the original spacing between elements and refine it
    d = original_list[1] - original_list[0]
    d_refined = d / factor
        
    max_value = max(original_list)
    min_value = min(original_list)
    
    # the number of elements is the number of divisions between
    # the limit values, plus one aditional
    n_refined = ((max_value - min_value) / d_refined) + 1
        
    return np.linspace(min_value,max_value,math.ceil(n_refined))


def refine_yticks(ax,factor):
    """
    Take an Axis object 'ax' and refine the y-ticks on it by 'factor'
    
    Parameters
    ----------
    ax : matplotlib.pyplot.Axex.Axis() object 
    factor: int
    Returns
    -------
    out : None 
    """

    ax.set_yticks(refine_list(ax.get_yticks(),factor),minor=True)

def refine_xticks(ax,factor):
    """
    Take an Axis object 'ax' and refine the x-ticks on it by 'factor'
    
    Parameters
    ----------
    ax : matplotlib.pyplot.Axex.Axis() object 
    factor: int
    Returns
    -------
    out : None 
    """

    ax.set_xticks(refine_list(ax.get_xticks(),factor),minor=True)
    

def create_plot(xlabel="",
                     ylabel="",
                     title=""):
    """
    Return (fig,axis) correspondent to a line plot,
    with 'xlabel','ylabel' and 'title'
    
    The size of the figure is controlled by FIGSIZE_INCHES"""
    
    fig = plt.figure()
    axis = fig.add_subplot(111)
    
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    
    return fig, axis

def create_two_axes_plot(xlabel="", 
                              ylabel_left="", 
                              ylabel_right="",
                              title=""):
    """
    Return (fig, axis_left, axis_right) correspondent
    to a line plot with two y-axes.
    
    The size of the figure is controlled by FIGSIZE_INCHES
    """
    
    fig, axis_left = create_plot(xlabel, ylabel_left, title)
    
    axis_right = axis_left.twinx()
    axis_right.set_ylabel(ylabel_right)
    
    return (fig, axis_left, axis_right)
    
def save_figure(fig,name):
    """
    Save the 'fig' Figure object as 'name' (with extension PLOT_EXTENSION),
    inside FIG_FILE_PATH (a Path-like object)."""
    
    fig_path = FIG_FILE_PATH
    file_basename = name + PLOT_EXTENSION
    
    file_path = fig_path / file_basename
    fig.savefig(str(file_path),
                dpi=DPI,
                bbox_inches='tight')

# In[3]:



def filter_table_from_column(table,column,value):
    """
    Return a view into the 'table' DataFrame, selecting only the rows where
    'column' equals 'value'
    """
    
    return table[table[column] == value]

# Fixed parameters

FIXED_PARAMETERS = {
    "D_p[m]": 0.5e-3,
    "L[m]": 100e-3,
    "W[m]": 25e-3,
    "H[m]": 20e-3,
    "N_r[]": 11,
    "T_H[K]": 298,
    "dT[K]": 20}

# ## 3 Comparison of profiles
# 
# For the instantaneous profile, the blow fraction is 1.0, while for the cosine profile the blow fraction is 0.6. 
# This is to compare the best situations testes for both profiles.
# 
# The $x$-axis shows the maximum value for the instantaneous profile and the average during the cold blow for the cosine profile 
# (i.e. the average during the entire blow period, not the average when there is actual fluid flow).

def plot_Qc_and_COP_Inst_vs_CCH(table_inst,table_cch,F_inst,F_CCH, figure_suffix=""):
    """
    Plots figures of Qc x B and COP x B from data from the DataFrames 'table_inst' and 'table_cch',
    selecting points with blow fractions 'F_inst' and 'F_CCH',
    and adding 'figure_suffix' to the end of the filename for each figure
    
    Creates one figure for each value of f, with curves for constant Phi for each profile
    """ 

    regsim_utilizations = {0.1988: 0.2,
                          0.3977: 0.4,
                          0.5965: 0.6,
                          0.7953: 0.8,
                          0.9942: 1.0}
    
    f_vector = table_inst[ld.FREQUENCY_COLUMN].unique()
    phi_vector = table_inst[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()    
    H_max_vector = table_inst[ld.MAXIMUM_PROFILE_COLUMN].unique()
                    
    x_min = min(H_max_vector)
    x_max = max(H_max_vector)

    fig_list = []
    markers=['s','o','x','v','^','h','<']
    for f in f_vector:    
        table_inst_f = filter_table_from_column(table_inst,ld.FREQUENCY_COLUMN,f)
        table_cch_f = filter_table_from_column(table_cch,ld.FREQUENCY_COLUMN,f)
           
        table_inst_f_F = filter_table_from_column(table_inst_f,ld.BLOW_FRACTION_COLUMN,F_inst) 
        table_cch_f_F = filter_table_from_column(table_cch_f,ld.BLOW_FRACTION_COLUMN,F_CCH)
        
        
        #Para os casos com mesmo minimo e mesma media
        if  table_inst_f_F[ld.MINIMUM_PROFILE_COLUMN].max() <= 0.1:
            table_inst_f_F_Hmin = filter_table_from_column(table_inst_f_F,ld.MINIMUM_PROFILE_COLUMN,0.1) 
            
        if  table_inst_f_F[ld.MINIMUM_PROFILE_COLUMN].max() > 0.1:
            table_inst_f_F_Hmin = table_inst_f_F   
                       
                     
        fig_Qc, axis = create_plot(ylabel=Q_LABEL,
                                               xlabel=B_LABEL)
        
        fig_COP, axis_COP = create_plot(ylabel=COP_LABEL,
                                               xlabel=B_LABEL)
                
        y_max = 0
        y_COP_max = 0
                             
                  
        for (i,phi) in enumerate(phi_vector):

            label_text = r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) + r'$'

            table_inst_f_F_Hmin_phi = filter_table_from_column(table_inst_f_F_Hmin,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)
            table_cch_f_F_phi = filter_table_from_column(table_cch_f_F,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)

            x_vector = table_inst_f_F_Hmin_phi[ld.MAXIMUM_PROFILE_COLUMN].values
            Qc_vector_inst = table_inst_f_F_Hmin_phi[ld.COOLING_CAPACITY_COLUMN].values
            Qc_vector_cch = table_cch_f_F_phi[ld.COOLING_CAPACITY_COLUMN].values
            
            COP_vector_inst = table_inst_f_F_Hmin_phi[ld.COP_COLUMN].values
            COP_vector_cch = table_cch_f_F_phi[ld.COP_COLUMN].values

                    
            axis.plot(x_vector, Qc_vector_inst, label='IT '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) 
                      + r'$', 
                      marker=markers[i],
                      linestyle='-',
                      color='k')
            axis.plot(x_vector, Qc_vector_cch, label='RC '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) + r'$', 
                      marker=markers[i],
                      linestyle='--', 
                      color='k')
            axis.legend(loc='upper left',
                   bbox_to_anchor=(1.0,1.0))
            
            axis_COP.plot(x_vector, COP_vector_inst, label='IT '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) 
                      + r'$',                       
                    marker=markers[i],
                      linestyle='-',
                      color='k')
            axis_COP.plot(x_vector, COP_vector_cch, label='RC '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) + r'$', 
                        marker=markers[i],
                      linestyle='--',
                      color='k')
            axis_COP.legend(loc='upper left',
                   bbox_to_anchor=(1.0,1.0))
                    
            if (max(Qc_vector_inst) > y_max):
                y_max = max(Qc_vector_inst)
                    
            if (max(COP_vector_inst) > y_COP_max):
                y_COP_max = max(COP_vector_inst)
        
        axis.set_ylim(0,y_max)     
        axis.set_xlim(x_min,x_max)
        axis.grid(True)
        refine_xticks(axis,4)
        refine_yticks(axis,4)
        
        axis_COP.set_ylim(0,y_COP_max)     
        axis_COP.set_xlim(x_min,x_max)
        axis_COP.grid(True)
        refine_xticks(axis_COP,4)
        refine_yticks(axis_COP,4)
        
        fig_list.append((fig_Qc,fig_COP))
        fig_Qc_name = "Qc_B_comp_f_%d%s" %(f,figure_suffix)
        save_figure(fig_Qc,fig_Qc_name)
        fig_COP_name = "COP_B_comp_f_%d%s" %(f,figure_suffix)
        save_figure(fig_COP,fig_COP_name)


# ### Same minimum
# 
# In this analysis, the minimum value of the cosine and the instantaneous profile are the same (the situation represented by the profile figure above)

# In[24]:

table_inst = ld.itdf
table_cch = ld.rcdf

F_B_inst = 100
F_B_CCH = 60
fig_suffix = "_same_minimum"
plot_Qc_and_COP_Inst_vs_CCH(
    table_inst, 
    table_cch,
    F_B_inst,
    F_B_CCH,
    fig_suffix)

def plot_Wpump_Inst_vs_CCH(table_inst,table_cch,F_inst,F_CCH, figure_suffix=""):
    """
    Plots figures of Wpump x B from data from the DataFrames 'table_inst' and 'table_cch',
    selecting points with blow fractions 'F_inst' and 'F_CCH',
    and adding 'figure_suffix' to the end of the filename for each figure
    
    Creates one figure for each value of f, with curves for constant Phi for each profile
    """ 

    regsim_utilizations = {0.1988: 0.2,
                          0.3977: 0.4,
                          0.5965: 0.6,
                          0.7953: 0.8,
                          0.9942: 1.0}
    
    f_vector = table_inst[ld.FREQUENCY_COLUMN].unique()
    phi_vector = table_inst[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()    
    H_max_vector = table_inst[ld.MAXIMUM_PROFILE_COLUMN].unique()
                    
    x_min = min(H_max_vector)
    x_max = max(H_max_vector)

    fig_list = []
    markers=['s','o','x','v','^','h','<']
    for f in f_vector:    
        table_inst_f = filter_table_from_column(table_inst,ld.FREQUENCY_COLUMN,f)
        table_cch_f = filter_table_from_column(table_cch,ld.FREQUENCY_COLUMN,f)
           
        table_inst_f_F = filter_table_from_column(table_inst_f,ld.BLOW_FRACTION_COLUMN,F_inst) 
        table_cch_f_F = filter_table_from_column(table_cch_f,ld.BLOW_FRACTION_COLUMN,F_CCH)
        
        
        #Para os casos com mesmo minimo e mesma media
        if  table_inst_f_F[ld.MINIMUM_PROFILE_COLUMN].max() <= 0.1:
            table_inst_f_F_Hmin = filter_table_from_column(table_inst_f_F,ld.MINIMUM_PROFILE_COLUMN,0.1) 
            
        if  table_inst_f_F[ld.MINIMUM_PROFILE_COLUMN].max() > 0.1:
            table_inst_f_F_Hmin = table_inst_f_F   
                       
                     
        fig, axis = create_plot(ylabel=W_PUMP_LABEL,
                                               xlabel=B_LABEL)
        
                
        y_max = 0
        y_COP_max = 0
                             
                  
        for (i,phi) in enumerate(phi_vector):

            label_text = r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) + r'$'

            table_inst_f_F_Hmin_phi = filter_table_from_column(table_inst_f_F_Hmin,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)
            table_cch_f_F_phi = filter_table_from_column(table_cch_f_F,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)

            x_vector = table_inst_f_F_Hmin_phi[ld.MAXIMUM_PROFILE_COLUMN].values
            Wpump_vector_inst = table_inst_f_F_Hmin_phi[ld.PUMPING_POWER_COLUMN].values
            Wpump_vector_cch = table_cch_f_F_phi[ld.PUMPING_POWER_COLUMN].values

                    
            axis.plot(x_vector, Wpump_vector_inst, label='IT '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) 
                      + r'$', 
                      marker=markers[i],
                      linestyle='-',
                      color='k')
            axis.plot(x_vector, Wpump_vector_cch, label='RC '+r'$\Phi = ' + '%.1f' %(regsim_utilizations[phi]) + r'$', 
                      marker=markers[i],
                      linestyle='--', 
                      color='k')
            axis.legend(loc='upper left',
                   bbox_to_anchor=(1.0,1.0))
            

                    
            if (max(Wpump_vector_inst) > y_max):
                y_max = max(Wpump_vector_inst)
                    
        
        axis.set_ylim(0,y_max)     
        axis.set_xlim(x_min,x_max)
        axis.grid(True)
        refine_xticks(axis,4)
        refine_yticks(axis,4)
        
        
        fig_list.append((fig,))
        fig_name = "Wpump_B_comp_f_%d%s" %(f,figure_suffix)
        #save_figure(fig,fig_name)
   
    return fig_list

plot_Wpump_Inst_vs_CCH(table_inst, table_cch,F_B_inst,F_B_CCH,fig_suffix)


# ### Instantaneous profile -  Q_c vs $\Phi$ (one curve for each maximum field)

# In[28]:


def plot_Qc_phi_Inst(table):
    """
    Plots figures of Qc x Phi from data from the DataFrame 'table'
    
    Creates one figure for each combination of (f, H_min, F_B), with curves for constant H_max
    """

    regsim_utilizations = {0.1988: 0.2,
                          0.3977: 0.4,
                          0.5965: 0.6,
                          0.7953: 0.8,
                          0.9942: 1.0}
    
    f_vector = table[ld.FREQUENCY_COLUMN].unique()
    phi_vector = table[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()
    F_blow_vector = table[ld.BLOW_FRACTION_COLUMN].unique()
    H_max_vector = table[ld.MAXIMUM_PROFILE_COLUMN].unique()
    H_min_vector = table[ld.MINIMUM_PROFILE_COLUMN].unique()

    fig_list = []
    
    markers=['s','o','x','v','^','h','<']
    for f in f_vector:    
        table_f = filter_table_from_column(table,ld.FREQUENCY_COLUMN,f)
        
        for F_blow in F_blow_vector:
            table_f_F = filter_table_from_column(table_f,ld.BLOW_FRACTION_COLUMN,F_blow)
            
            for H_min in H_min_vector:
                table_f_F_Hmin = filter_table_from_column(table_f_F,ld.MINIMUM_PROFILE_COLUMN,H_min)     
                     
                fig_Qc, axis_Qc = create_plot(xlabel=PHI_LABEL,
                                                       ylabel=Q_LABEL)
                
                fig_COP, axis_COP = create_plot(xlabel=PHI_LABEL,
                                                       ylabel=COP_LABEL)
   
                for (i,Hmax) in enumerate(H_max_vector):


                    table_f_F_Hmin_Hmax = filter_table_from_column(table_f_F_Hmin,ld.MAXIMUM_PROFILE_COLUMN,Hmax)

                    x_vector = table_f_F_Hmin_Hmax[ld.UTILIZATION_HOT_BLOW_COLUMN].values
                    Qc_vector = table_f_F_Hmin_Hmax[ld.COOLING_CAPACITY_COLUMN].values
                    COP_vector = table_f_F_Hmin_Hmax[ld.COP_COLUMN].values

                    axis_Qc.plot(x_vector, Qc_vector, 
                              label=r'$\overline{B}_\mathrm{max} = ' + '%.1f' %(Hmax,) + r'\mathrm{T}$',
                              marker=markers[i],linestyle='--',color='k'  )

                    
                    axis_COP.plot(x_vector, COP_vector, 
                              label=r'$\overline{B}_\mathrm{max} = ' + '%.1f' %(Hmax,) + r'\mathrm{T}$',
                              marker=markers[i],linestyle='--',color='k'  )
                    
                    
                for ax in [axis_Qc, axis_COP]:
                    
                    ax.plot(x_vector,np.zeros_like(x_vector),
                                color='grey',
                                linestyle='-')
                    
                    ax.legend(loc='upper left',
                             bbox_to_anchor=(1,1))    
                    ax.set_xlim(0.2,1)
                    ax.grid(True)
                    refine_yticks(ax,5)
              
                fig_list.append((fig_Qc,fig_COP))
                
                
                save_figure(fig=fig_Qc,
                                    name='Qc_Phi_inst_f_%d_Hmin_%03d_FB_%d' %(f,100*H_min,F_blow))
                plt.close(fig_Qc)
                save_figure(fig=fig_COP,
                                    name='COP_Phi_inst_f_%d_Hmin_%03d_FB_%d' %(f,100*H_min,F_blow))
            
                plt.close(fig_COP)

plot_Qc_phi_Inst(table_inst)

# # ### Instantaneous profile -  Q_c vs H_reg 
# # 
# # - Width, length and number of regenerators are kept fixed
# # - Blow fraction is kept fixed at 100%
# # - Minimum field 0.05 T

# # In[30]:


def plot_Qc_H_Inst(table):
    """
    Plots figures of Qc x H_reg from data from the DataFrame 'table'
    
    Creates one figure for each combination of (f, Phi), with curves for constant H_max
    """

    regsim_utilizations = {0.1988: 0.2,
                          0.3977: 0.4,
                          0.5965: 0.6,
                          0.7953: 0.8,
                          0.9942: 1.0}
    
    f_vector = table[ld.FREQUENCY_COLUMN].unique()
    phi_vector = table[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()
    H_vector = table['H[mm]'].unique()
    H_max_vector = table[ld.MAXIMUM_PROFILE_COLUMN].unique()
    
    x_min = min(H_vector)
    x_max = max(H_vector)
    
    fig_list = []
    markers=['s','o','x','v','^','h','<']
    for f in f_vector:    
        table_f = filter_table_from_column(table,ld.FREQUENCY_COLUMN,f)
        
        for phi in phi_vector:
            
            table_f_phi = filter_table_from_column(table_f,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)
            
                     
            fig_Qc, axis_Qc = create_plot(xlabel=H_REG_LABEL,
                                                       ylabel=Q_LABEL)
             
            fig_COP, axis_COP = create_plot(xlabel=H_REG_LABEL,
                                                       ylabel=COP_LABEL)
            
            for (i, H_max) in enumerate(H_max_vector):
                
                table_f_phi_Hmax = filter_table_from_column(table_f_phi,ld.MAXIMUM_PROFILE_COLUMN,H_max)

                x_vector = table_f_phi_Hmax['H[mm]'].values
                Qc_vector = table_f_phi_Hmax[ld.COOLING_CAPACITY_COLUMN].values
                COP_vector = table_f_phi_Hmax[ld.COP_COLUMN].values
                
                axis_Qc.plot(x_vector, Qc_vector, 
                             label=r'$\overline{B}_\mathrm{max} = ' + '%.1f' %(H_max,) + r'\ \mathrm{T}$',
                             marker=markers[i],linestyle='--',color='k'  )
                
                axis_COP.plot(x_vector, COP_vector, 
                              label=r'$\overline{B}_\mathrm{max} = ' + '%.1f' %(H_max,) + r'\ \mathrm{T}$',
                              marker=markers[i],linestyle='--',color='k'  )
                    
                
            for ax in [axis_Qc, axis_COP]:
                   
                ax.plot(x_vector,np.zeros_like(x_vector),
                        color='grey',
                         linestyle='-')
                    
                ax.legend(loc='upper left',
                         bbox_to_anchor=(1,1))    
                
                ax.grid(True)
                refine_yticks(ax,5)
                ax.set_xlim(min(x_vector),max(x_vector))
                ax.set_xticks(x_vector)
              
            fig_list.append((fig_Qc,fig_COP))
                
            save_figure(fig=fig_Qc,
                               name='Qc_H_inst_f_%d_Phi_%d' %(f,100*regsim_utilizations[phi]))
            plt.close(fig_Qc)
            save_figure(fig=fig_COP,
                                name='COP_H_inst_f_%d_Phi_%d' %(f,100*regsim_utilizations[phi]))
            plt.close(fig_COP)                           
# In[31]:

table_Inst_variosH = ld.it_varH_df
plot_Qc_H_Inst(table_Inst_variosH)


# # ## 4 Plotting the ramp profile

# # In[32]:


# def calculate_ramp_profile(phi, B_low, B_high, field_fraction):
#     """
#     Calculate the value of the two-pole ramp magnetic profile at angular position 'phi' (in degrees),
#     where the profile oscillates from 'B_low' to 'B_high', with a field fraction of 'field_fraction' 
#     (fraction of the cycle where the field is at its highest)
    
#     """

#     tau = 360
#     tau_M = field_fraction * tau
    
#     tau_R = 1.0/4 * (tau - 2*tau_M)
    
#     tan_theta_r = (B_high - B_low) / (2*tau_R)
        
#     initial_ascent_region = (phi < tau_R )
    
#     high_region = np.logical_and((phi >= tau_R ),(phi <= (tau/2 - tau_R)))

#     descent_region = np.logical_and((phi > (tau/2 - tau_R) ),(phi < (tau/2 + tau_R)))
    
    
#     final_ascent_region = (phi > (tau - tau_R))

                 
#     return np.where(initial_ascent_region,
#                    (B_high+B_low)/2 + phi * tan_theta_r,
#                     np.where(high_region,
#                     B_high,
#                    np.where(descent_region,
#                            B_high - (phi - (tau/2 - tau_R))*tan_theta_r,
#                            np.where(final_ascent_region,
#                                    B_low + (phi - (tau - tau_R))*tan_theta_r,
#                                    B_low))))


# # ## AMR curve

# # - Fixed regenerator
# # - Fixed span
# # - Vary frequency, utilization, ramp and blow fraction
# # - Magn. Period = Demagn. Period
# # - High field Period = Low field Period
# # - Magn Period + High field Period = 50%
# # 
# # The period the AMR cycle if $\tau$, divided into a cold stage period $\tau_C$ and a hot stage period $\tau_H$, such that $\tau_c = \tau_H$.
# # 
# # The blow fraction $F_B$ is the fraction of the entire cycle where there is fluid blow in a given AMR bed. The cold blow period is $\tau_{CB}$ and the how blow period is $\tau_{HB}$. Because of the symmetry between the cold and hot stages:
# # 
# # \begin{equation}
# # \tau_{CB} = F_B \tau_C = \frac{1}{2} F_B \tau = \tau_{HB}
# # \end{equation}
# # 
# # The high field fraction $F_M$ is the fraction of the entire cycle where the magnetic profile stays  fully magnetized (or, due to symmetry, fully demagnetized). 

# # In[33]:


# def plot_Qc_Rampa(table, figure_suffix=""):
#     """
#     Plots figures of Qc x F_M and COP x F_M from data from the DataFrame 'table',
#     adding 'figure_suffix' to the end of the filename for each figure
    
#     Creates one figure for each combination of (f, Phi), with curves for constant F_b
#     """

#     regsim_utilizations = {0.1988: 0.2,
#                           0.2982: 0.3, 
#                           0.3977: 0.4,
#                           0.4971: 0.5, 
#                           0.5965: 0.6,
#                           0.7953: 0.8,
#                           0.9942: 1.0}

    
#     f_vector = table[ld.FREQUENCY_COLUMN].unique()
#     phi_vector = table[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()
#     F_blow_vector = table[ld.BLOW_FRACTION_COLUMN].unique()    
#     Mag_Period = table['MagPeriod [s]'].unique()
    
#     markers=['s','o','x','v']
    

#     fig_list = []
#     fig_list_COP = []
#     for f in f_vector:    
        
#         table_f = filter_table_from_column(table,ld.FREQUENCY_COLUMN,f)
        
#         for phi in phi_vector:
            
#             table_f_phi = filter_table_from_column(table_f,ld.UTILIZATION_HOT_BLOW_COLUMN,phi) 
                
                     
#             fig_Qc, axis = create_plot(xlabel=RATIO_F_LABEL,
#                                                 ylabel=Q_LABEL)
            
#             fig_COP, axis_COP = create_plot(xlabel=RATIO_F_LABEL,
#                                                     ylabel=COP_LABEL+' [-]')
                                              
#             y_max = 0
                        
#             i = 0
#             for F_blow in F_blow_vector:

#                 label_text = r'$F_\mathrm{B} = ' + '%d' %(F_blow,) + r'{\%}$'

#                 table_f_phi_F = filter_table_from_column(table_f_phi,ld.BLOW_FRACTION_COLUMN,F_blow)

#                 x_vector = table_f_phi_F['FieldPeriod[s]'].values
                
#                 F_M_vector = 2 * x_vector
                
#                 ratio_vector = F_M_vector / F_blow
                
#                 Qc_vector = table_f_phi_F[ld.COOLING_CAPACITY_COLUMN].values
#                 COP_vector = table_f_phi_F[ld.COP_COLUMN].values

#                 axis.plot(ratio_vector, Qc_vector, label=label_text,marker=markers[i],linestyle='--',color='k')
#                 axis.legend(loc='best',fontsize=nemplot_parameters["FONTSIZE"])
                
#                 axis_COP.plot(ratio_vector, COP_vector, label=label_text,marker=markers[i],linestyle='--',color='k')
#                 axis_COP.legend(loc='best',fontsize=nemplot_parameters["FONTSIZE"])
#                 i = i + 1
                
#                 for ax in [axis,axis_COP]:
#                     ax.grid(True)
#                     #ax.set_xticks(ratio_vector)
#                     refine_yticks(ax,6)
#                     refine_xticks(ax,5)
          
                
#             fig_list.append(fig_Qc)
#             fig_list_COP.append(fig_COP)
#             save_figure(fig=fig_Qc,
#                                 name='Qc_FM_ramp_f_%d_Phi_%d%s' %(f,100*regsim_utilizations[phi],figure_suffix))
#             save_figure(fig=fig_COP,
#                                 name='COP_FM_ramp_f_%d_Phi_%d%s' %(f,100*regsim_utilizations[phi],figure_suffix))
        
        
#     return fig_list,fig_list_COP


# # In[34]:


# # Fixed parameters

# FIXED_PARAMETERS_NEW = {
#     "D_p[m]": 0.35e-3,
#     "L[m]": 85e-3,
#     "W[m]": 25e-3,
#     "H[m]": 22e-3,
#     "N_r[]": 8,
#     "T_H[K]": 305.5,
#     "dT[K]": 35,
#     "Casing material": "Stainless steel",
#     "t_casing[mm]": 0.5e-3,
#     "t_air[mm]": 1e-3,
#     "N_layers[]": 3,
#     "T_C_layers[K]": np.array([273,283,290]),
#     "Length_fraction_layers[%]": np.array([20,20,60]),
#     "B_min[T]": 0.05}


# # In[35]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# table_filename = 'Slope - Perdas -35K-valv_ASCO.txt'

# table = pd.read_csv(table_filename,sep='\t')

# table_13 = filter_table_from_column(table,'Hmax[T]',1.3)

# plot_Qc_Rampa(table_13,"_35K_1300mT");


# # ## 5 2D maps

# # In[36]:


# #criando tabela para o plot 2D
# table_filename12 = 'Slope - Perdas-1.2.txt'
# table_filename13 = 'Slope - Perdas-1.3.txt'
# table_filename14 = 'Slope - Perdas-1.4.txt'

# table_slope_2D = pd.read_csv(table_filename12,sep='\t')

# #código feito para retirar pontos do campo alto de 1.3T, pois alguns só existem para esse campo
# table = (pd.read_csv(table_filename13,sep='\t'))
# table1 = filter_table_from_column(table,ld.FREQUENCY_COLUMN,1) #Retirando freq 2

# table2 = table1[table1[ld.UTILIZATION_HOT_BLOW_COLUMN] == 0.1988] #Retirando utilização 1.0
# table2 = table2.append(table1[table1[ld.UTILIZATION_HOT_BLOW_COLUMN] == 0.3977])
# table2 = table2.append(table1[table1[ld.UTILIZATION_HOT_BLOW_COLUMN] == 0.5965])
# table2 = table2.append(table1[table1[ld.UTILIZATION_HOT_BLOW_COLUMN] == 0.7953])

# table3 = table2[table2[ld.BLOW_FRACTION_COLUMN] == 60]#Retirando 100
# table3 = table3.append(table2[table2[ld.BLOW_FRACTION_COLUMN] == 70])
# table3 = table3.append(table2[table2[ld.BLOW_FRACTION_COLUMN] == 80])
# table3 = table3.append(table2[table2[ld.BLOW_FRACTION_COLUMN] == 90])

# table4 = table3[table3['MagPeriod [s]'] == 5] #Retirando 0
# table4 = table4.append(table3[table3['MagPeriod [s]'] == 10])
# table4 = table4.append(table3[table3['MagPeriod [s]'] == 15])
# table4 = table4.append(table3[table3['MagPeriod [s]'] == 20])
# table4 = table4.append(table3[table3['MagPeriod [s]'] == 25])
# table4 = table4.append(table3[table3['MagPeriod [s]'] == 30])

# table_slope_2D = table_slope_2D.append(table4)
# table_slope_2D = table_slope_2D.append(pd.read_csv(table_filename14,sep='\t'))


# # In[37]:


# def plot_slope_2D(table_slope_2D,figure_suffix=""):
#     """
#     Plots 2D maps of  Q_c as a function of the high field level and high field fraction,
#     using from data from the DataFrame 'table',
#     adding 'figure_suffix' to the end of the filename for each figure
    
#     Creates one figure for each combination of (f, Phi, F_b)
#     """
    
#     regsim_utilizations = {0.1988: 0.2,
#                           0.2982: 0.3, 
#                           0.3977: 0.4,
#                           0.4971: 0.5, 
#                           0.5965: 0.6,
#                           0.7953: 0.8,
#                           0.9942: 1.0}
        
#     f_vector = table_slope_2D[ld.FREQUENCY_COLUMN].unique()
#     phi_vector = table_slope_2D[ld.UTILIZATION_HOT_BLOW_COLUMN].unique()
#     F_blow_vector = table_slope_2D[ld.BLOW_FRACTION_COLUMN].unique()    
#     Mag_Period = table_slope_2D['MagPeriod [s]'].unique()
#     HMax_vector = table_slope_2D['Hmax[T]'].unique()
#     FieldPeriod_vector = table_slope_2D['FieldPeriod[s]'].unique()

#     Qc_matrix = np.zeros((len(HMax_vector),len(FieldPeriod_vector)))
#     COP_matrix = Qc_matrix.copy()
    
#     fig_list = []
#     table_list = []
#     get_ipython().run_line_magic('matplotlib', 'inline')
#     for f in f_vector:
        
#         table_f = filter_table_from_column(table_slope_2D,ld.FREQUENCY_COLUMN,f)
#         for phi in phi_vector:
            
#             table_f_phi = filter_table_from_column(table_f,ld.UTILIZATION_HOT_BLOW_COLUMN,phi)                       
#             for F_blow in F_blow_vector:

                        
#                 fig_Qc, ax_Qc = create_plot(xlabel=F_M_LABEL,
#                                                     ylabel=H_MAX_LABEL)
                
#                 fig_COP, ax_COP = create_plot(xlabel=F_M_LABEL,
#                                                     ylabel=H_MAX_LABEL)
                
#                 table_f_phi_F = filter_table_from_column(table_f_phi,ld.BLOW_FRACTION_COLUMN,F_blow)

#                 Qc_vector = table_f_phi_F[ld.COOLING_CAPACITY_COLUMN].values
#                 COP_vector = table_f_phi_F[ld.COP_COLUMN].values
#                 #transformando o vetor Qc em matriz
#                 cont = 0
#                 for j in range(len(HMax_vector)):
#                     for i in range(len(FieldPeriod_vector)):                    
#                         Qc_matrix[j,i] = Qc_vector[cont]
#                         COP_matrix[j,i] = COP_vector[cont]
#                         cont += 1

#                 F_M_matrix,B_max_matrix = np.meshgrid(FieldPeriod_vector,HMax_vector)
#                 F_M_matrix = 2*F_M_matrix
#                 p_Qc = ax_Qc.contour(F_M_matrix,B_max_matrix,Qc_matrix,colors='k')
#                 ax_Qc.clabel(p_Qc, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%1.0f')
#                 refine_xticks(ax_Qc,5)
#                 refine_yticks(ax_Qc,5)
                    
#                 p_COP = ax_COP.contour(F_M_matrix,B_max_matrix,COP_matrix,colors='k')
#                 ax_COP.clabel(p_COP, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%.1f')
#                 refine_xticks(ax_COP,5)
#                 refine_yticks(ax_COP,5)
                
#                 ax_Qc.grid(True)
#                 ax_COP.grid(True)
           
#                 table_list.append(table_f_phi_F)
#                 fig_list.append(fig_Qc)
#                 fig_list.append(fig_COP)
#                 save_figure(fig_Qc,
#                                     name='Qc_ramp_map_f_%d_Phi_%d_FB_%d%s' %(f,
#                                                                           100*regsim_utilizations[phi],
#                                                                           F_blow,
#                                                                           figure_suffix))
#                 plt.close(fig_Qc)
#                 save_figure(fig_COP,
#                                     name='COP_ramp_map_f_%d_Phi_%d_FB_%d%s' %(f,
#                                                                           100*regsim_utilizations[phi],
#                                                                           F_blow,
#                                                                           figure_suffix))
#                 plt.close(fig_COP)
    
#     return fig_list, table_list            


# # In[38]:


# table_slope_2D_35K = pd.read_csv('Slope - Perdas -35K-valv_ASCO.txt',sep='\t')
# fig_list_slope2D_35K, table_list_slope2D_35K = plot_slope_2D(table_slope_2D_35K,figure_suffix='_35K_Valv_ASCO')


# # ### 6.2 Variando altura do regenerador

# # Todos os outros parâmetros fixos. Fração de magnetização 70%.

# # In[39]:


# def plot_slope_multipleH(table_slope_2D,figure_suffix=""):    
    
#     HMax_vector = table_slope_2D['Hmax[T]'].unique()
#     H_vector = table_slope_2D['H[mm]'].unique()

#     Qc_matrix = np.zeros((len(HMax_vector),len(H_vector)))
#     COP_matrix = Qc_matrix.copy()
#     eta_matrix = Qc_matrix.copy()
    
#     fig_list = []
#     table_list = []

#     fig_Qc, ax_Qc = create_plot(title='',
#                                         xlabel=H_REG_LABEL,
#                                         ylabel=H_MAX_LABEL)

#     fig_COP, ax_COP = create_plot(title='',
#                                         xlabel=H_REG_LABEL,
#                                         ylabel=H_MAX_LABEL)
    
#     fig_eta, ax_eta = create_plot(title='',
#                                         xlabel=H_REG_LABEL,
#                                         ylabel=H_MAX_LABEL)
    
#     fig_combined, ax_combined = create_plot(title='',
#                                         xlabel=H_REG_LABEL,
#                                         ylabel=H_MAX_LABEL)


    
#     Qc_vector = table_slope_2D[ld.COOLING_CAPACITY_COLUMN].values
#     COP_vector = table_slope_2D[ld.COP_COLUMN].values
    
#     T_H = FIXED_PARAMETERS_NEW["T_H[K]"]
#     dT_sys = FIXED_PARAMETERS_NEW["dT[K]"]
#     T_C = T_H - dT_sys
    
#     eta_vector = COP_vector *  dT_sys / T_C * 100 # the factor of 100 is to transform to percentage
    
#     #transformando o vetor Qc em matriz
#     cont = 0
#     for i in range(len(H_vector)):
#         for j in range(len(HMax_vector)):                            
#             Qc_matrix[j,i] = Qc_vector[cont]
#             COP_matrix[j,i] = COP_vector[cont]
#             eta_matrix[j,i] = eta_vector[cont]
#             cont += 1

#     X,Y = np.meshgrid(H_vector,HMax_vector)
#     p_Qc = ax_Qc.contour(X,Y,Qc_matrix,colors='k')
#     ax_Qc.clabel(p_Qc, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%1.0f')
#     refine_xticks(ax_Qc,5)
#     refine_yticks(ax_Qc,5)
            
#     p_COP = ax_COP.contour(X,Y,COP_matrix,colors='k')
#     ax_COP.clabel(p_COP, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%.1f')
#     refine_xticks(ax_COP,5)
#     refine_yticks(ax_COP,5)

#     p_eta = ax_eta.contour(X,Y,eta_matrix,colors='k')
#     ax_eta.clabel(p_eta, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%.1f')
#     refine_xticks(ax_eta,5)
#     refine_yticks(ax_eta,5)
    
#     p_combined_Qc = ax_combined.contour(X,Y,Qc_matrix,5,colors='k',linestyles='-')
#     ax_combined.clabel(p_combined_Qc, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%.1f')
#     p_combined_COP = ax_combined.contour(X,Y,COP_matrix,5,colors='k',linestyles='--')
#     ax_combined.clabel(p_combined_COP, fontsize=0.8*nemplot_parameters["FONTSIZE"], inline=1,fmt='%.1f')

#     ax_Qc.grid(True)
#     ax_COP.grid(True)
#     ax_eta.grid(True)
#     ax_combined.grid(True)
    
#     refine_xticks(ax_combined,5)
#     refine_yticks(ax_combined,5)

#     fig_list.append((fig_Qc,fig_COP,fig_eta,fig_combined))
    
#     save_figure(fig_Qc,
#                         name='Qc_H_reg%s' %(
#                             figure_suffix,))
#     plt.close(fig_Qc)
    
#     save_figure(fig_COP,
#                         name='COP_H_reg%s' %(figure_suffix,))
#     plt.close(fig_COP)
    
#     save_figure(fig_eta,
#                         name='eta_H_reg%s' %(figure_suffix,))
#     plt.close(fig_eta)
    
#     save_figure(fig_combined,
#                         name='Qc_COP_combined_H_reg%s' %(figure_suffix,))
#     plt.close(fig_combined)


   
#     return fig_list, table_list  


# # In[40]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# table_slope_variosH = pd.read_csv('Slope - Perdas - 35K - Varios H - Parte 2.txt',sep='\t')
# fig_list_slope2D_35K = plot_slope_multipleH(table_slope_variosH,figure_suffix='W30')


plt.close('all')