#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pandas import DataFrame, Series

INSTANTANEOUS_DATASET = "Instantaneous.txt"
INSTANTANEOUS_DATASET_VARYING_HEIGHT = "Instantaneous-varH.txt"
RAMP_DATASET = "Ramp.txt"
RECTIFIED_COSINE_DATASET = "RectifiedCosine.txt"


# These dataset are formated as tab-separed values (TSV), with column names in the format:
# 
#     <column_name>[<unit>]

# These files are formatted in such a way that they can be easily loaded with pandas DataFrames.
# 
# We adopt the convention of naming these data sets as '<profile_identifier>df', where the profile identifier can be 'it' (instantaneous), 'rc' (rectified cosine) or 'rm' (ramp)

itdf = pd.read_csv(INSTANTANEOUS_DATASET,sep="\s+")
it_varH_df = pd.read_csv(INSTANTANEOUS_DATASET_VARYING_HEIGHT,sep="\s+")
rcdf = pd.read_csv(RECTIFIED_COSINE_DATASET,sep="\s+")
rmdf = pd.read_csv(RAMP_DATASET,sep="\s+")


# The columns of interest can be ly identified as:

FREQUENCY_COLUMN = 'f[Hz]'
MAXIMUM_PROFILE_COLUMN = 'H_max[T]'
MINIMUM_PROFILE_COLUMN = 'H_min[T]'
BLOW_FRACTION_COLUMN  = 'F_B[%]'
UTILIZATION_HOT_BLOW_COLUMN = 'U_HB[-]'
UTILIZATION_COLD_BLOW_COLUMN = 'U_CB[-]'
PRESSURE_DROP_COLD_BLOW_COLUMN =  'dPCB[kPa]'
PRESSURE_DROP_HOT_BLOW_COLUMN =  'dPHB[kPa]'
SYSTEM_TEMPERATURE_SPAN_COLUMN = 'Tspan[K]'
REGENERATOR_TEMPERATURE_SPAN_COLUMN = 'dT_reg[K]'
COOLING_CAPACITY_COLUMN =  'Qc[W]'
HEAT_REJECTION_COLUMN = 'Qh[W]'
PUMPING_POWER_COLUMN =  'Wpump[W]'
VALVE_POWER_COLUMN = 'Wvalv[W]'
MAGNETIC_POWER_COLUMN = 'Wmotor[W]'
COP_COLUMN = 'COP[-]'
RAMP_FRACTION_COLUMN = 'F_R[%]'
HIGH_MAGNETIZATION_FRACTION_COLUMN =  'F_M_High[%]'
REYNOLDS_COLD_BLOW_COLUMN = 'ReDp_CB[-]'
REYNOLDS_HOT_BLOW_COLUMN = 'ReDp_HB[-]'
HEAT_LOSS_WALL_COLUMN = 'Q_wall-Loss[W]'

