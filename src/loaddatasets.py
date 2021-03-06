#!/usr/bin/env python
# coding: utf-8

"""
loaddatasets.py

Read the csv files and load them as pandas DataFrames.
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

SRC_DIR = Path(__file__).resolve().parent

ROOT_DIR = SRC_DIR.parent.resolve()

DATA_DIR = ROOT_DIR/ "data"

INSTANTANEOUS_DATASET = "Instantaneous.csv"
INSTANTANEOUS_DATASET_VARYING_HEIGHT = "Instantaneous-varH.csv"
RAMP_DATASET = "Ramp.csv"
RAMP_DATASET_VARYING_HEIGHT = "Ramp-varH.csv"
RECTIFIED_COSINE_DATASET = "RectifiedCosine.csv"


# These dataset are formated as tab-separed values (TSV), with column names in the format:
# 
#     <column_name>[<unit>]

# These files are formatted in such a way that they can be easily loaded with pandas DataFrames.
# 
# We adopt the convention of naming these data sets as '<profile_identifier>df', where the profile identifier can be 'it' (instantaneous), 'rc' (rectified cosine) or 'rm' (ramp)

itdf = pd.read_csv(DATA_DIR / INSTANTANEOUS_DATASET)
it_varH_df = pd.read_csv(DATA_DIR / INSTANTANEOUS_DATASET_VARYING_HEIGHT)
rcdf = pd.read_csv(DATA_DIR / RECTIFIED_COSINE_DATASET)
rmdf = pd.read_csv(DATA_DIR / RAMP_DATASET)
rm_varH_df = pd.read_csv(DATA_DIR / RAMP_DATASET_VARYING_HEIGHT)


# The columns of interest can be nicely identified as:

FREQUENCY_COLUMN = 'f[Hz]'
REGENERATOR_HEIGHT_COLUMN = 'H[mm]'
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

