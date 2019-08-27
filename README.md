# Paper: Influence of the magnetic field profile on the performance of active magnetic regenerators

This repository contain datasets, scripts and manuscript files for our group's paper on the subject explained below.

Part of this analysis was already published in:

- Fortkamp, Fábio P., Lang, Gusttav B., Lozano, Jaime A., & Barbosa Jr., Jader R. (2017). Performance of magnetic refrigerators operating with different magnetic profiles. *Proceedings of the 24th ABCM International Congress of Mechanical Engineering (COBEM 2017)*. Curitiba.
- Fortkamp, Fábio Pinto (2019). Integrated Design of the Magnet-Regenerator Assembly for a Magnetic Refrigerator. PhD Thesis (Department of Mechanical Engineering). Federal University of Santa Catarina, Florianópolis.

## Datasets files

Three dataset files are provided with steady state results for an AMR configuration. The magnetic profiles used are identified by the filename.  Different test conditions were used for the different profiles. 

The AMR model is described in detail in:

- Trevizoli, P. V. (2015). Development of thermal tegenerators for magnetic cooling applications. PhD Thesis (Department of Mechanical Engineering), Federal University of Santa Catarina, Florianópolis.

and was implemented in the RegSim software.

### Common parameters for `Instantaneous.csv` and `RectifiedCosine.csv`

For the instantaneous and rectified cosine profiles, the common AMR parameters used are:

- 11 Prismatic regenerators with rectangular cross section
- Gd spheres as the magnetocaloric material
- Porosity of 36%
- Particle diameter of 0.5 mm
- Regenerator height of 20 mm
- Regenerator width of 25 mm
- Regenerator length of 100 mm
- Hot source temperature of 298 K
- Cold source temperature of 278 K

In addition, a digital hydraulic system was used with the following parameters:

- 22 valves
- Nominal power for one valve of 4 W (the so-called "Type B" valve)
- Nominal power for one controlling relay of 0.36 W (and one relay can be used to control 2 valves)

The simulations for these profiles also considered adiabatic regenerators.

### Common parameters for `Instantaneous-varH.csv"

This dataset contains results for simulation of the instantaneous magnetic profile with varying regenerator height.

The following parameters are kept fixed:

- Blow fraction of 100 %
- Minimum magnetic field of 0.05 T

The remaining conditions are as the `Instantaneous.csv` dataset.

### Common parameters for `Ramp.csv`

The simulations for the ramp magnetic profile dataset also considered:

- Casing losses
- Multilayer renegerators (consisted of "shifted" Gd layers)
- Low-consumption valves ("Type A" valves), whose power depends on frequency and blow fraction

Common parameters:

- 8 regenerators and 16 valves
- Regenerator width of 25 mm (the value used in Fortkamp's thesis was actually wrong)
- Regenerator height of 22 mm
- Regenerator length of 85 mm
- Casing thickness of 0.5 mm, composed of stainless steel
- Air layer (around the casing) with thickness of 1 mm
- Cold side temperature of 270.5 K
- Hot side temperature of 305.5 K
- Particle diameter of 350 um
- Regenerators with three layers, with Curie temperatures of {273, 283, 290} K and respective length fractions of {20, 20, 60} % of the regenerator length

In addition to the canonical `Ramp.csv`, there are also files entitled `Ramp-BmaxXXXXmT.csv`, with results with the maximum field fixed at the value indicated by `XXXX` (e.g. 1200 mT = 1.2 T).

### Common parameters for `Ramp-varH.csv`

This dataset contains simulations of the ramp magnetic profile with varying regenerator height. The following parameters are fixed:

- Regenerator width of 30 mm
- Magnetization fraction of 70 % (considering both high and low field plateaus)
- Blow fraction identical to the magnetization fraction


### Columns

The columns name for the `Instantaneous.csv` and `Instantaneous-varH.csv` datasets are as follows (the units are indicated between square brackets):

* `f[Hz]`: AMR cycle frequency
* `H[mm]`: regenerator height
* `H_max[T]`: maximum applied field (actually is $\mu_0 H$, where $\mu_0$ is the magnetic permeability of free space)
* `H_min[T]`: minimum applied field
* `F_B[%]`: blow fraction
* `U_HB[-]`: hot blow utilization factor
* `dPCB[kPa]`: cold blow pressure drop through one regenerator
* `Tspan[K]`: system temperature span (equivalent to regenerator span, since this data set assumes ideal heat exchangers)
* `Qc[W]`: cooling capacity
* `Qh[W]`: heat rejection rate
* `Wpump[W]`: pumping power
* `Wvalv[W]`: valve consumption 	
* `Wmotor[W]`: motor/magnetic power (assumed identical to the Carnot ideal power)
* `COP[-]`: coefficient of performance

For the `RectifiedCosine.csv` dataset, there is an additional column, `H_max_equiv[T]`, which represents the equivalent instantaneous magnetic profile with the same average field *during the entire half-cycle period*, i.e. this value is not dependent on blow fraction. In addition for this dataset, the minimum applied field is set fixed to 0.1 T.

The files `Ramp-Bmax.csv`, in addition of the columns shown above, also has the following fields:

* `F_R[%]`: fraction of the cycle where the magnetic field is changing (the "ramp fraction")
* `F_M_High[%]`: fraction of the cycle where the magnetic field is at its maximum level
* `ReDp_CB[-]`: particle diameter-based Reynolds number during cold blow
* `ReDp_HB[-]`: particle diameter-based Reynolds number during hot blow
* `U_CB[-]`: cold blow utilization factor
* `dPHB[kPa]`: hot blow pressure drop through one regenerator
* `dT_reg[K]`: regenerator temperature span (and `Tspan[K]` then represents the system temperature span)
* `Q_wall-Loss[W]`: heat leakage through the regenerators wall

The identifiers `Test` and `Re_w[-]` can be ignored.

## Python scripts

The figures for the paper are generated via Python scripts. To reproduce them, it is recommended that you create a conda environment with the specified conda file:

    conda env create -f environment.yml

The environment file specifies the versions of the softwares used to create the plots.

The scripts are purposed as follows:

* `plotprofiles.py`: generates figures illustrating the different profiles waveforms

The scripts are set to be saved on disk so that they can be loaded into the LaTeX manuscript.