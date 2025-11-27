# AI Assistant Guidance

This file provides guidance to my AI assistant when working with code in this repository.  In addition there is a report from my AI assistant about the IceCube Kaggle Competition that is relevant to our work.

## Project Overview

This is a fork of GraphNeT which we are using to build a GNN for direction reconstruction at MAGIC. While the GraphNeT codebase is important, we will focus on developing and training GNN networks.

## MAGIC and Available Data Overview and Context
We have raw data for a pair of IACT (cherenkov) telescopes called MAGIC.

### Data Format

- Each node contains the following features
    - pixel X and Y position (standardized between [-1, +1]
    - signal (~n of photoelectrons in the pixel), standardized with an asinh function to preserve information across orders of magnitude
    - time t at which the signal arrived, standardized. 
    - telescope ID (as 0 or 1)
- We have two graph level parameters (which are currently passed to every node as a feature) are the phi and theta values of the telescope pointing (which is usually ~0.4deg offset from source position on the sky. These are also standardized
- The cameras have hexagonal pixels so each pixel generally has 6 neighbors each
- In each of the two telescopes there are 1039 pixels across 50 slices in time. In theory, this could mean up to 51950 nodes / telescopes
- However, we apply a cleaning algorithm (standard two-level cleaning with core/boundary pixels) that also considers neighbors in the time dimension. In general there are about 200-2000 nodes per graph, so much more managable
- The graphs are currently structured such that XYZ is formed with X,Y,t. The t dimension is scaled such that the intra-timeslice distance is equal to the intra-pixel distance on the camera (but this can be adjusted)
- We are using graphs instead of CNNs because a) not-standard pixel shape (hexagons!) and b) timing is different in each pixel, so camera snapshots are not aligned in time.
- We are using the graphnet framework, a code that is made for reconstructing neutrino events with icecube. We will be the first ones using it for gamma rays!

### What do we want to reconstruct?

- energy, arrival direction of particle (theta and phi), and binary classification [photon or hadron]
- energy is logged and normalized using a QuantileTransformer to correct for the power-law distribution of energies in our simulations
- Dataset:
    - 75000 gamma ray events
    - 75000 proton events
- simulated at various zenith angles between 5 and 35º below zenith. samples at each angle is proportional to the deg2 size of the sky at that angle (i.e. many more events for 30-35deg compared to 5-10deg)
- simulated for energies between 10GeV and 100TeV. samples at each energy scale with a powerlaw (log space) since low-energy photons are much more common
- Tasks to complete:
    - Classification— classify events in a binary way between gamma ray and proton.
    - Energy reconstruction— predict the energy for gamma rays only
    - Direction reconstruction— predict the location on the sky (zenith and azimuth) from which the gamma ray arrived, predict only for gamma rays
- Note that the tasks should communicate in some way (i.e. energy and direction need only be computed for gamma rays!)

### Memory: Code Editing Guidelines
- try not to edit files in the graphnet codebase (/src) that are not ours. it's better to create new files. __init__ files an similaar things are ok!

### Memory: Environment Activation
- pleaase use the full micromamba python envirnoment path before every python command like so `/home/iwsatlas1/jgreen/micromamba/envs/graphnet-121-v3/bin/python {ORIGINL COMMAND}`

[... rest of the existing file content remains unchanged ...]
