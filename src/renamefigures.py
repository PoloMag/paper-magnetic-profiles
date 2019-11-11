"""
renamefigures.py

Rename figures that will appear in the manuscript, with a 
"FigX-" prefix, and move these copies to the root dir.
"""
from pathlib import Path
from shutil import copy2
from string import ascii_lowercase

from plotprofiles import FIGURES_DIR, ROOT_DIR

def main():
    figures_list = [
        "reg3d.eps",
        "mprofile.eps",
        "profiles_it_and_rc.eps",
        "profile_rm.eps",
        "profiles_it_and_rc_same_minimum.eps",
        "profiles_rc_and_flow_instantaneous.eps",
        "Qc_B_comp_f_1_same_minimum.eps",
        "COP_B_comp_f_1_same_minimum.eps",
        "W_B_comp_f_1_Phi_60_same_minimum.eps",
        (
            "Qc_Phi_inst_f_1_Hmin_005_FB_100.eps",
            "Qc_Phi_inst_f_2_Hmin_005_FB_100.eps",
        ),
        (
            "Qc_H_inst_f_1_Phi_40.eps",
            "COP_H_inst_f_1_Phi_40.eps"
        ),
        "amr-casing.eps",
        "profiles_rm_and_flow_instantaneous.eps",
        (
            "Qc_FM_ramp_f_1_Phi_40_35K_1300mT.eps",
            "COP_FM_ramp_f_1_Phi_40_35K_1300mT.eps"
        ),
        (
            "Qc_FM_ramp_f_1_Phi_20_35K_1300mT.eps",
            "Qc_FM_ramp_f_1_Phi_50_35K_1300mT.eps"
        ),
        (
            "COP_ramp_map_f_1_Phi_20_FB_60_35K_Valv_ASCO.eps",
            "COP_ramp_map_f_1_Phi_20_FB_90_35K_Valv_ASCO.eps",
            "COP_ramp_map_f_1_Phi_40_FB_60_35K_Valv_ASCO.eps",
            "COP_ramp_map_f_1_Phi_40_FB_90_35K_Valv_ASCO.eps"
        ),
        (
            "Qc_COP_combined_H_regW30.eps",
            "eta_H_regW30.eps"
        )
    ]

    for i, f in enumerate(figures_list):

        if not isinstance(f,tuple):
            p = Path(FIGURES_DIR / f)
            dst_filename = "Fig%d-%s" %(i+1,f)
            dst_path = ROOT_DIR / dst_filename
            copy2(str(p),str(dst_path))
        else:
            for j, subf in enumerate(f):
                p = Path(FIGURES_DIR / subf)
                dst_filename = "Fig%d%s-%s" %(
                    i+1,
                    ascii_lowercase[j],
                    subf)
                dst_path = ROOT_DIR / dst_filename
                copy2(str(p),str(dst_path))

if __name__ == "__main__":
    main()