import numpy as np

import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt

N_POINTS_LINEPLOT = 2000

FIGSIZE_IN = (8,6)
DPI = 800

# stolen from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('text', usetex=True)
plt.rc(
    'font', 
    size=MEDIUM_SIZE,
    family='serif',
    serif='Helvetica')          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE,figsize=FIGSIZE_IN)  # fontsize of the figure title

def calculate_instantaneous_profile(phi, B_low, B_high):
    """
    Calculate the value of the two-pole instantaneous magnetic profile at angular position 'phi' (in degrees),
    where the profile oscillates from 'B_low' to 'B_high'
    
    """
    
    high_region = (phi < 180)
                 
    return np.where(high_region,B_high,B_low)

def calculate_rectified_cosine_profile(phi, B_low, B_high):
    """
    Calculate the value of the two-pole rectified cosine magnetic profile at angular position 'phi' (in degrees),
    where the profile oscillates from 'B_low' to 'B_high'
    
    """
    
    return B_low + (B_high-B_low)*np.abs(np.cos((np.deg2rad(phi)-np.pi/2)/2))

def calculate_ramp_profile(phi, B_low, B_high, field_fraction):
    """
    Calculate the value of the two-pole ramp magnetic profile at angular position 'phi' (in degrees),
    where the profile oscillates from 'B_low' to 'B_high', with a field fraction of 'field_fraction' 
    (fraction of the cycle where the field is at its highest)
    
    """

    tau = 360
    tau_M = field_fraction * tau
    
    tau_R = 1.0/4 * (tau - 2*tau_M)
    
    tan_theta_r = (B_high - B_low) / (2*tau_R)
        
    initial_ascent_region = (phi < tau_R )
    
    high_region = np.logical_and((phi >= tau_R ),(phi <= (tau/2 - tau_R)))

    descent_region = np.logical_and((phi > (tau/2 - tau_R) ),(phi < (tau/2 + tau_R)))
    
    
    final_ascent_region = (phi > (tau - tau_R))

                 
    return np.where(initial_ascent_region,
                   (B_high+B_low)/2 + phi * tan_theta_r,
                    np.where(high_region,
                    B_high,
                   np.where(descent_region,
                           B_high - (phi - (tau/2 - tau_R))*tan_theta_r,
                           np.where(final_ascent_region,
                                   B_low + (phi - (tau - tau_R))*tan_theta_r,
                                   B_low))))

def calculate_flow_instantaneous_profile(phi, m_max, blow_fraction):
    """
    Calculate the value of the instantaneous flow profile at angular position 'phi' (in degrees),
    where the profile oscillates from -'m_max' to 'm_max', with a blow fraction of 'blow_fraction' 
    """

    tau = 360
    tau_B = blow_fraction * tau/2
    
    tau_0 = (tau/2 - tau_B)/2
    
    high_region = np.logical_and((phi >= tau_0 ),(phi <= (tau/2 - tau_0)))

    low_region = np.logical_and((phi >= (tau/2 + tau_0) ),(phi <= (tau - tau_0)))
    
    return np.where(high_region,
                    m_max,
                   np.where(low_region,
                           -m_max,0))


def plot_all_profiles():
    "Return a figure with all profiles."
    B_max= 1.0
    B_min = 0.1
    F_M = 0.4

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_cos = calculate_rectified_cosine_profile(time_deg,B_min,B_max)
    B_apl_inst = calculate_instantaneous_profile(time_deg,B_min,B_max)
    B_apl_ramp = calculate_ramp_profile(time_deg,B_min,B_max,F_M)

    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_cos,'k-',label="Rectified Cosine")
    profile_axes.plot(time,B_apl_inst,'k--',label="Instantaneous")
    profile_axes.plot(time,B_apl_ramp,'k-.',label="Ramp")
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,(B_max+B_min)/2,B_max])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'$\frac{\left({B}_\mathrm{max}+{B}_\mathrm{min}\right)}{2}$',
                                  r'${B}_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'$B$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.legend(loc='upper left',
                       bbox_to_anchor=(1.0,1.0))

    profile_axes.set_ylim(0,1.1*B_max)

    tau_M = F_M * tau
    tau_R = 1.0/4 * (tau - 2*tau_M)

    # it seems the way to create a bidiretional arrow is to create an "annotation" with an empty text,
    # and using the 'arrowprops' dictionary
    profile_axes.annotate("", xy=(tau_R,1.02*B_max), xytext=(tau_R+tau_M,1.02*B_max), arrowprops=dict(arrowstyle='<->'))

    profile_axes.text(0.99*tau/4,1.035*B_max,r'$\tau_\mathrm{M}$',
                      horizontalalignment='right')

    profile_fig.savefig("profiles_all.eps",dpi=800,bbox_inches="tight")

def plot_it_and_rc_profiles():
    B_max= 1.0
    B_min = 0.1
    F_M = 0.4

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_cos = calculate_rectified_cosine_profile(time_deg,B_min,B_max)
    B_apl_inst = calculate_instantaneous_profile(time_deg,B_min,B_max)

    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_cos,'k-',label="Rectified Cosine")
    profile_axes.plot(time,B_apl_inst,'k--',label="Instantaneous")
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,B_max])
    profile_axes.set_yticklabels([r'$B_\mathrm{min}$',
                                  r'$B_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'$B$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.legend(loc='upper left',
                       bbox_to_anchor=(1.0,1.0))

    profile_axes.set_ylim(0,1.1*B_max)

    profile_fig.savefig("profiles_it_and_rc.eps",dpi=DPI,bbox_inches='tight')

def plot_it_and_rc_profiles_same_minimum():

    B_max_ins= 1.0
    B_max_cos = 1.1
    B_min = 0.1

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_cos = calculate_rectified_cosine_profile(time_deg,B_min,B_max_cos)
    B_apl_inst = calculate_instantaneous_profile(time_deg,B_min,B_max_ins)

    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_cos,'k-',label="Rectified Cosine")
    profile_axes.plot(time,B_apl_inst,'k--',label="Instantaneous")
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,B_max_ins,])
    profile_axes.set_yticklabels([r'$B_\mathrm{min}$',
                                  r'$\overline{B}_\mathrm{high}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'$B$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.legend(loc='upper left',
                       bbox_to_anchor=(1.0,1.0))
    B_max = max(B_max_cos,B_max_ins)
    profile_axes.set_ylim(0,1.2*B_max)

    profile_fig.savefig("profiles_it_and_rc_same_minimum.eps",dpi=DPI,bbox_inches="tight")

def plot_it_and_rc_profiles_same_average():
    B_max_ins= 1.0
    B_min_ins = 0.473
    B_max_cos = 1.1
    B_min_cos = 0.1

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_cos = calculate_rectified_cosine_profile(time_deg,B_min_cos,B_max_cos)
    B_apl_inst = calculate_instantaneous_profile(time_deg,B_min_ins,B_max_ins)

    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_cos,'k-',label="Rectified Cosine")
    profile_axes.plot(time,B_apl_inst,'k--',label="Instantaneous")
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min_cos,B_min_ins,B_max_ins])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'$\overline{B}_\mathrm{low}$',
                                  r'$\overline{B}_\mathrm{high}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.legend(loc='upper left',
                       bbox_to_anchor=(1.0,1.0))

    B_max = max(B_max_cos,B_max_ins)
    profile_axes.set_ylim(0,1.2*B_max)

    profile_fig.savefig("profiles_it_and_rc_same_average.eps",dpi=DPI,bbox_inches="tight")

def plot_it_profile():
    B_max= 1.0
    B_min = 0.1
    F_M = 0.4

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)


    B_apl_inst = calculate_instantaneous_profile(time_deg,B_min,B_max)


    profile_fig, profile_axes = plt.subplots()


    profile_axes.plot(time,B_apl_inst,'k-',label="Instantaneous")

    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,B_max])
    profile_axes.set_yticklabels([r'$B_\mathrm{min}$',
                                  r'$B_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)


    profile_axes.set_ylim(0,1.1*B_max)

    profile_fig.savefig("profile_it.eps",dpi=DPI,bbox_inches="tight")

def plot_rc_profile():
    B_max= 1.0
    B_min = 0.1
    F_M = 0.4

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_cos = calculate_rectified_cosine_profile(time_deg,B_min,B_max)


    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_cos,'k-',label="Rectified Cosine")

    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,B_max])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'${B}_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.set_ylim(0,1.1*B_max)

    profile_fig.savefig("profile_rc.eps",dpi=DPI,bbox_inches="tight")

def plot_rm_profile():
    B_max= 1.0
    B_min = 0.1
    F_M = 0.4

    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_ramp = calculate_ramp_profile(time_deg,B_min,B_max,F_M)

    profile_fig, profile_axes = plt.subplots()

    profile_axes.plot(time,B_apl_ramp,'k-',label="Ramp")
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,(B_max+B_min)/2,B_max])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'$\frac{\left({B}_\mathrm{max}+{B}_\mathrm{min}\right)}{2}$',
                                  r'${B}_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    #profile_axes.legend(loc='best',fontsize=0.8*nemplot_parameters["FONTSIZE"])

    profile_axes.set_ylim(0,1.1*B_max)

    tau_M = F_M * tau
    tau_R = 1.0/4 * (tau - 2*tau_M)

    # it seems the way to create a bidiretional arrow is to create an "annotation" with an empty text,
    # and using the 'arrowprops' dictionary
    profile_axes.annotate("", xy=(tau_R,1.02*B_max), xytext=(tau_R+tau_M,1.02*B_max), arrowprops=dict(arrowstyle='<->'))

    profile_axes.text(0.99*tau/4,1.035*B_max,r'$\tau_\mathrm{M}$',
                      horizontalalignment='right')

    # add annotation for the ramp rate
    profile_axes.text(0.9*tau/2,1.4*B_min,r'$\theta_\mathrm{R}$',
                      horizontalalignment='center')

    xcenter_arc = tau/2 + tau_R
    ycenter_arc = B_min
    radius_arc = tau_R

    angle_arc = 0
    theta_1_arc = 90 + 1.5*np.rad2deg(np.arctan(tau_R))
    theta_2_arc = 180

    arc = patches.Arc((xcenter_arc, ycenter_arc), 2*radius_arc, radius_arc,
                    angle=angle_arc,
                      theta1=theta_1_arc,
                      theta2=theta_2_arc,
                     linewidth=1, 
                      linestyle='dashed',
                     fill=False, 
                     zorder=2)

    profile_axes.add_patch(arc)

    profile_axes.annotate("", xy=(3*tau/8,B_min), xytext=(tau/2+tau_R,B_min) ,arrowprops=dict(arrowstyle='-',linestyle='--'))

    profile_fig.savefig("profile_rm.eps",dpi=DPI,bbox_inches="tight")

def plot_rc_and_flow_instantaneous_profiles():
    B_max= 1.0
    B_min = 0.1
    m_max = 1.0
    F_B = 0.6
    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_ramp = calculate_rectified_cosine_profile(time_deg, B_min, B_max)
    m_inst = calculate_flow_instantaneous_profile(time_deg,m_max,F_B)

    profile_fig, profile_axes = plt.subplots()
    profile_axes_right = profile_axes.twinx()

    profile_axes.plot(time,B_apl_ramp,'k-')
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,B_max])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'${B}_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.set_ylim(-1.3*B_max,1.3*B_max)

    profile_axes_right.plot(time,m_inst,'k--')
    profile_axes_right.set_yticks([-m_max,0,m_max])
    profile_axes_right.set_yticklabels([r'$-\dot{m}_\mathrm{f,max}$',
                                  r'0',
                                  r'$\dot{m}_\mathrm{f,max}$'])

    profile_axes_right.set_ylabel(r'$\dot{m}_\mathrm{f}$',
                           rotation='horizontal')
    profile_axes_right.set_ylim(-1.3*B_max,1.3*B_max)
    profile_axes_right.yaxis.grid(True)
    profile_axes_right.yaxis.set_label_coords(1.05,1.05)

    profile_fig.savefig("profiles_rc_and_flow_instantaneous.eps",dpi=DPI,bbox_inches="tight")

def plot_rm_and_flow_instantaneous_profile():

    B_max= 1.0
    B_min = 0.1
    F_M = 0.4
    m_max = 0.8
    F_B = 0.7
    tau = 2*np.pi

    time = np.linspace(0,tau,N_POINTS_LINEPLOT)
    time_deg = np.rad2deg(time)

    B_apl_ramp = calculate_ramp_profile(time_deg,B_min,B_max,F_M)
    m_inst = calculate_flow_instantaneous_profile(time_deg,m_max,F_B)

    profile_fig, profile_axes = plt.subplots()
    profile_axes_right = profile_axes.twinx()

    profile_axes.plot(time,B_apl_ramp,'k-')
    profile_axes.set_xlim(0,np.max(time))
    profile_axes.set_xticks(np.linspace(0,np.max(time),5))
    profile_axes.xaxis.grid(True)
    profile_axes.set_xticklabels([r'$0$', r'$\frac{\tau}{4}$', r'$\frac{\tau}{2}$',r'$\frac{3\tau}{4}$',r'$\tau$'])

    profile_axes.set_yticks([B_min,(B_max+B_min)/2,B_max])
    profile_axes.set_yticklabels([r'${B}_\mathrm{min}$',
                                  r'$\frac{\left({B}_\mathrm{max}+{B}_\mathrm{min}\right)}{2}$',
                                  r'${B}_\mathrm{max}$'])
    profile_axes.yaxis.grid(True)

    profile_axes.set_xlabel(r'$t$')

    profile_axes.set_ylabel(r'${B}$',
                           rotation='horizontal')

    profile_axes.xaxis.set_label_coords(1.04,-0.02)
    profile_axes.yaxis.set_label_coords(-0.03,1.0)

    profile_axes.set_ylim(-1.3*B_max,1.3*B_max)

    tau_M = F_M * tau
    tau_R = 1.0/4 * (tau - 2*tau_M)

    # it seems the way to create a bidiretional arrow is to create an "annotation" with an empty text,
    # and using the 'arrowprops' dictionary
    profile_axes.annotate("", xy=(tau_R,1.07*B_max), xytext=(tau_R+tau_M,1.07*B_max), arrowprops=dict(arrowstyle='<->'))

    profile_axes.text(0.99*tau/4,1.12*B_max,r'$\tau_\mathrm{M}$',
                      horizontalalignment='right')

    profile_axes_right.plot(time,m_inst,'k--')
    profile_axes_right.set_yticks([-m_max,0,m_max])
    profile_axes_right.set_yticklabels([r'$-\dot{m}_\mathrm{f,max}$',
                                  r'0',
                                  r'$\dot{m}_\mathrm{f,max}$'])

    profile_axes_right.set_ylabel(r'$\dot{m}_\mathrm{f}$',
                           rotation='horizontal')
    profile_axes_right.set_ylim(-1.3*B_max,1.3*B_max)
    profile_axes_right.yaxis.grid(True)
    profile_axes_right.yaxis.set_label_coords(1.05,1.05)

    tau_B = F_B * tau/2
    tau_0 = 1.0/4 * (tau - 2*tau_B)

    # it seems the way to create a bidiretional arrow is to create an "annotation" with an empty text,
    # and using the 'arrowprops' dictionary
    profile_axes_right.annotate("", xy=(tau/2+tau_0,-1.1*m_max), xytext=(tau - tau_0,-1.1*m_max), arrowprops=dict(arrowstyle='<->'))
    profile_axes_right.text(2.99*tau/4,-1.3*m_max,r'$\tau_\mathrm{B}$',
                      horizontalalignment='right')


    profile_fig.savefig("profiles_rm_and_flow_instantaneous.eps",dpi=DPI,bbox_inches="tight")

plot_all_profiles()
plot_it_and_rc_profiles()
plot_it_and_rc_profiles_same_average()
plot_it_and_rc_profiles_same_minimum()
plot_it_profile()
plot_rc_profile()
plot_rm_profile()
plot_rc_and_flow_instantaneous_profiles()
plot_rm_and_flow_instantaneous_profile()

plt.close('all')