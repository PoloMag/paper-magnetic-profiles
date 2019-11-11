# Paper: Numerical analysis of the influence of magnetic field waveforms on the performance of active magnetic regenerators

Authors: Fábio P. Fortkamp, Gusttav B. Lang, Jaime A. Lozano, Jader R. Barbosa Jr, all from the Federal University of Santa Catarina, Brazil.

Submitted to the [Journal of the Brazilian Society of Mechanical Sciences and Engineering](https://www.springer.com/journal/40430?detailsPage=pltci_2297590).

This repository contain datasets, scripts and manuscript files for our group's paper on the subject explained in the title.

Parts of this analysis were already published in:

- Fortkamp, Fábio P., Lang, Gusttav B., Lozano, Jaime A., & Barbosa Jr., Jader R. (2017). Performance of magnetic refrigerators operating with different magnetic profiles. *Proceedings of the 24th ABCM International Congress of Mechanical Engineering (COBEM 2017)*. Curitiba.
- Fortkamp, Fábio Pinto (2019). Integrated Design of the Magnet-Regenerator Assembly for a Magnetic Refrigerator. PhD Thesis (Department of Mechanical Engineering). Federal University of Santa Catarina, Florianópolis.

## Datasets files

Dataset files in CSV format, inside the `data` directory, are provided with steady state results for an AMR configuration. The magnetic profiles used are identified by the filename.  Different test conditions were used for the different profiles, as explained in the sections below. The columns of these spreadsheets represent the model variables, and each row is a different simulation, with different input parameters (such as amplitude and frequency of the magnetic profile, utilization factor, regenerator geometry) and the resulting output values (pressure drop, power contributions, cooling capacity, coefficient of performance).

The AMR model is described in detail in:

- Trevizoli, P. V. (2015). Development of thermal tegenerators for magnetic cooling applications. PhD Thesis (Department of Mechanical Engineering), Federal University of Santa Catarina, Florianópolis.

and was implemented in the RegSim software developed by the author above. Unfortunately, we cannot make this software available; the numerical values output by RegSim in its results files were manually inserted into the spreadsheets explained below.

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
- Nominal power for one valve of 4 W (the so-called "Type R" valve)
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
- Low-consumption valves ("Type S" valves), whose power depends on frequency and blow fraction

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

The identifiers `Test` and `Re_w[-]` can be ignored.

## Python scripts

The figures for the paper are generated via Python scripts contained in the `src` directory; the figures are output to the `figures` directory. To reproduce them, it is recommended that you create a conda environment with the specified conda file:

    conda env create -f environment.yml

The environment file specifies the versions of the softwares used to create the plots.

The scripts are purposed as follows:

* `loaddatasets.py`: reads the tables from the `data` directory and loads them as pandas DataFrames
* `plotprofiles.py`: generates figures illustrating the different profiles waveforms
* `figures.py`: generate various types of plots from the datasets. This script was primarily developed by G. B. Lang.
* `renamefigures.py`: copy figures generated by the above scripts and save copies in the root directory of the paper, preprending figure numbers to the file names. These numbers are hardcoded into the script, reflecting the state of the manuscript before the first submission.

To generate and copy *all* figures, you can run:

    python src/runall.py

Notice that this driver script does not compile the manuscript.

## Manuscript

The LaTeX manuscript is `PaperMagneticProfiles_JBSMSE.tex`. It uses templates and instruction provided by the journal.  

The BibTeX files `Thermo-Foam-Ref-bib` and `thesis.bib` are bibliography databases I (F. P. Fortkamp) have been carrying and updating for a long time, so they contain multiple entries not cited in the paper. These files were copied to this folder from [this repository](https://github.com/PoloMag/thermo-ref) by our group.

The best way to compile is with `latexmk`; although a chain of `pdflatex` and `bibtex` commands should work. To compile the list of symbols with the [`nomencl`](https://ctan.org/pkg/nomencl?lang=en) package, the `makeindex` command should be configured as:

    makeindex PaperMagneticProfiles_JBSMSE.nlo -s nomencl.ist -o PaperMagneticProfiles_JBSMSE.nls