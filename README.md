# Generalized positivity constraint on magnetic equivalent layers

by
Andre L. A. Reis,
Vanderlei C. Oliveira Jr.,
Valeria C. F. Barbosa

This paper has been published in *Geophysics*.
The version of record

> Reis, A.L.A, Oliveira Jr., V. C., & Barbosa, V. C. F., 2020. 
> Generalized positivity constraint on magnetic equivalent layers, 
> *Geophysics*, doi: 10.1190/geo2019-0706.1.

is available online at: [10.1190/geo2019-0706.1](https://doi.org/10.1190/geo2019-0706.1)

**This repository contains the data and code used to produce all results and figures shown
in the paper.**
This version of this repository is available at
[pinga-lab/eqlayer-magnetization-direction](https://github.com/pinga-lab/eqlayer-magnetization-direction)

We present a novel methodology for estimating the magnetization direction of geological sources
using equivalent-layer technique. Moreover, we prove the existence of an equivalent layer having 
an all-positive magnetic-moment distribution for the case in which the magnetization direction of this 
layer is the same as that of the true sources.

![Montes Claros complex application](montes-claros-application.png)
*Application of the methodology to the Montes Claros complex of Goias Alkaline Province, center of Brazil.
(a) Predicted data of Montes Claros complex with estimated magnetization direction. (b)
The positive magnetic-moment distribution. (c) The figure of an equivalent layer representation.*

## Abstract

It is known from potential theory that a continuous and planar layer of dipoles 
can exactly reproduce the total-field anomaly produced by arbitrary 3D sources. 
We prove the existence of an equivalent layer having an all-positive 
magnetic-moment distribution for the case in which the magnetization direction 
of this layer is the same as that of the true sources, regardless of whether the 
magnetization of the true sources is purely induced or not. 
By using this generalized positivity constraint, we present a new iterative method 
for estimating the total magnetization direction of 3D magnetic sources based on 
the equivalent-layer technique. Our method does not impose a priori information 
either about the shape or depth of the sources, does not require regularly spaced 
data, and presumes that the sources have a uniform magnetization direction. 
At each iteration, our method performs two steps. The first one solves a 
constrained linear inverse problem to estimate a positive magnetic-moment 
distribution over a discrete equivalent layer of dipoles. We consider that the 
equivalent sources are located on a plane and have an uniform and fixed 
magnetization direction. In the second step we use the estimated magnetic-moment 
distribution and solve a nonlinear inverse problem for estimating a new 
magnetization direction for the dipoles. The algorithm stops when the equivalent 
layer yields a total-field anomaly that fits the observed data. 
Tests with synthetic data simulating different geological scenarios show that 
the final estimated magnetization direction is close to the true one. 
We apply our method to a field data from the Goiás Alkaline Province (GAP), 
over the Montes Claros complex, center of Brazil. The results suggest the presence 
of intrusions with remarkable remanent magnetization, in agreement with the current 
literature for this region.

## Reproducing the results

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/pinga-lab/eqlayer-magnetization-direction.git

or [click here to download a zip archive](https://github.com/pinga-lab/eqlayer-magnetization-direction/archive/master.zip).

All source code used to generate the results and figures in the paper are in
the `code` folder. There you can find the Python codes that performs 
the synthetic data calculations and scripts to generate all figures and results
presented in the paper.

The sources for the manuscript text and figures are in `manuscript`.

See the `README.md` files in each directory for a full description.


### Setting up your environment and dependencies

You'll need a working Python **2.7** environment with all the standard
scientific packages installed (numpy, scipy, matplotlib, etc).  The easiest
(and recommended) way to get this is to download and install the
[Anaconda Python distribution](http://continuum.io/downloads#all).
Make sure you get the **Python 2.7** version.

We use `conda` virtual environments to manage the project dependencies in
isolation. Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create

You'll also need to install version 0.5 of the
[Fatiando a Terra](http://www.fatiando.org/) library.
See the install instructions on the website.

## License

All source code is made available under a BSD 3-clause license.  You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors.  See `LICENSE.md` for the full license text.

Data and the results of numerical tests are available under the
[Creative Commons Attribution 4.0 License (CC-BY)](https://creativecommons.org/licenses/by/4.0/).

The manuscript text and figures are not open source. The authors reserve the
rights to the article content, which has been accepted for publication in
Geophysics.

