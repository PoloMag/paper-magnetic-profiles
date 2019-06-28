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

### Common parameters for `Instantaneous.txt` and `RectifiedCosine.txt`

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

### Common parameters for `Ramp.txt`

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