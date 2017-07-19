SEAICE Package
--------------

<!– CMIREDIR:package\_seaice: –>

Authors: Martin Losch, Dimitris Menemenlis, An Nguyen, Jean-Michel
Campin, Patrick Heimbach, Chris Hill and Jinlun Zhang

Introduction [sec:pkg:seaice:intro]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package “seaice” provides a dynamic and thermodynamic interactive
sea-ice model.

CPP options enable or disable different aspects of the package (Section
[sec:pkg:seaice:config]). Run-Time options, flags, filenames and
field-related dates/times are set in (Section [sec:pkg:seaice:runtime]).
A description of key subroutines is given in Section
[sec:pkg:seaice:subroutines]. Input fields, units and sign conventions
are summarized in Section [sec:pkg:seaice:fields\_units], and available
diagnostics output is listed in Section [sec:pkg:seaice:diagnostics].

SEAICE configuration, compiling & running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile-time options [sec:pkg:seaice:config]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 

As with all MITgcm packages, SEAICE can be turned on or off at compile
time

-  using the file by adding to it,

-  or using adding or switches

-  | *required packages and CPP options*:
   | SEAICE requires the external forcing package to be enabled; no
     additional CPP options are required.

(see Section [sec:buildingCode]).

Parts of the SEAICE code can be enabled or disabled at compile time via
CPP preprocessor flags. These options are set in . Table
[tab:pkg:seaice:cpp] summarizes the most important ones. For more
options see the default .

[tab:pkg:seaice:cpp]

+------------------+------------------------------------------------------------------------------------------------------------+
| **CPP option**   | **Description**                                                                                            |
+==================+============================================================================================================+
|                  | Enhance STDOUT for debugging                                                                               |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | sea-ice dynamics code                                                                                      |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | LSR solver on C-grid (rather than original B-grid)                                                         |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable use of EVP rheology solver                                                                          |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable use of JFNK rheology solver                                                                         |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | use EXF-computed fluxes as starting point                                                                  |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | use differentialable regularization for viscosities                                                        |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable linear dependence of the freezing point on salinity (by default undefined)                          |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable snow to ice conversion for submerged sea-ice                                                        |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable sea-ice with variable salinity (by default undefined)                                               |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | enable sea-ice tracer package (by default undefined)                                                       |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | B-grid only for backward compatiblity: turn on ice-stress on ocean                                         |
+------------------+------------------------------------------------------------------------------------------------------------+
|                  | B-grid only for backward compatiblity: use ETAN for tilt computations rather than geostrophic velocities   |
+------------------+------------------------------------------------------------------------------------------------------------+

Table: Some of the most relevant CPP preprocessor flags in the -package.

Run-time parameters [sec:pkg:seaice:runtime]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run-time parameters (see Table [tab:pkg:seaice:runtimeparms]) are set in
files (read in ), and (read in ).

Enabling the package
^^^^^^^^^^^^^^^^^^^^

|  
| A package is switched on/off at run-time by setting (e.g. for SEAICE)
  in .

General flags and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| Table [tab:pkg:seaice:runtimeparms] lists most run-time parameters.

Input fields and units[sec:pkg:seaice:fields\_units]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:
    Initial sea ice thickness averaged over grid cell in meters;
    initializes variable ;

:
    Initial fractional sea ice cover, range :math:`[0,1]`; initializes
    variable ;

:
    Initial snow thickness on sea ice averaged over grid cell in meters;
    initializes variable ;

:
    Initial salinity of sea ice averaged over grid cell in
    g/m\ :math:`^2`; initializes variable ;

Description [sec:pkg:seaice:descr]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[TO BE CONTINUED/MODIFIED]

The MITgcm sea ice model (MITgcm/sim) is based on a variant of the
viscous-plastic (VP) dynamic-thermodynamic sea ice model
:raw-latex:`\citep{zhang97}` first introduced by
:raw-latex:`\citet{hib79, hib80}`. In order to adapt this model to the
requirements of coupled ice-ocean state estimation, many important
aspects of the original code have been modified and improved
:raw-latex:`\citep{losch10:_mitsim}`:

-  the code has been rewritten for an Arakawa C-grid, both B- and C-grid
   variants are available; the C-grid code allows for no-slip and
   free-slip lateral boundary conditions;

-  three different solution methods for solving the nonlinear momentum
   equations have been adopted: LSOR :raw-latex:`\citep{zhang97}`, EVP
   :raw-latex:`\citep{hun97}`, JFNK
   :raw-latex:`\citep{lemieux10,losch14:_jfnk}`;

-  ice-ocean stress can be formulated as in
   :raw-latex:`\citet{hibler87}` or as in :raw-latex:`\citet{cam08}`;

-  ice variables are advected by sophisticated, conservative advection
   schemes with flux limiting;

-  growth and melt parameterizations have been refined and extended in
   order to allow for more stable automatic differentiation of the code.

The sea ice model is tightly coupled to the ocean compontent of the
MITgcm. Heat, fresh water fluxes and surface stresses are computed from
the atmospheric state and – by default – modified by the ice model at
every time step.

The ice dynamics models that are most widely used for large-scale
climate studies are the viscous-plastic (VP) model
:raw-latex:`\citep{hib79}`, the cavitating fluid (CF) model
:raw-latex:`\citep{fla92}`, and the elastic-viscous-plastic (EVP) model
:raw-latex:`\citep{hun97}`. Compared to the VP model, the CF model does
not allow ice shear in calculating ice motion, stress, and deformation.
EVP models approximate VP by adding an elastic term to the equations for
easier adaptation to parallel computers. Because of its higher accuracy
in plastic solution and relatively simpler formulation, compared to the
EVP model, we decided to use the VP model as the default dynamic
component of our ice model. To do this we extended the line successive
over relaxation (LSOR) method of :raw-latex:`\citet{zhang97}` for use in
a parallel configuration. An EVP model and a free-drift implemtation can
be selected with runtime flags.

Compatibility with ice-thermodynamics package [sec:pkg:seaice:thsice]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| Note, that by default the -package includes the orginial so-called
  zero-layer thermodynamics following :raw-latex:`\citet{hib80}` with a
  snow cover as in :raw-latex:`\citet{zha98a}`. The zero-layer
  thermodynamic model assumes that ice does not store heat and,
  therefore, tends to exaggerate the seasonal variability in ice
  thickness. This exaggeration can be significantly reduced by using
  :raw-latex:`\citeauthor{sem76}`’s [:raw-latex:`\citeyear{sem76}`]
  three-layer thermodynamic model that permits heat storage in ice.
  Recently, the three-layer thermodynamic model has been reformulated by
  :raw-latex:`\citet{win00}`. The reformulation improves model physics
  by representing the brine content of the upper ice with a variable
  heat capacity. It also improves model numerics and consumes less
  computer time and memory.

The Winton sea-ice thermodynamics have been ported to the MIT GCM; they
currently reside under . The package is described in
section [sec:pkg:thsice]; it is fully compatible with the packages and .
When turned on together with , the zero-layer thermodynamics are
replaced by the Winton thermodynamics. In order to use the -package with
the thermodynamics of , compile both packages and turn both package on
in ; see an example in . Note, that once is turned on, the variables and
diagnostics associated to the default thermodynamics are meaningless,
and the diagnostics of have to be used instead.

Surface forcing[sec:pkg:seaice:surfaceforcing]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| The sea ice model requires the following input fields: 10-m winds, 2-m
  air temperature and specific humidity, downward longwave and shortwave
  radiations, precipitation, evaporation, and river and glacier runoff.
  The sea ice model also requires surface temperature from the ocean
  model and the top level horizontal velocity. Output fields are surface
  wind stress, evaporation minus precipitation minus runoff, net surface
  heat flux, and net shortwave flux. The sea-ice model is global: in
  ice-free regions bulk formulae are used to estimate oceanic forcing
  from the atmospheric fields.

Dynamics[sec:pkg:seaice:dynamics]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  

The momentum equation of the sea-ice model is

.. math::

   \label{eq:momseaice}
     m \frac{D{\ensuremath{\vec{\mathbf{u}}}}}{Dt} = -mf{\ensuremath{\vec{\mathbf{k}}}}\times{\ensuremath{\vec{\mathbf{u}}}} + {{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{air} +
     {{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{ocean} - m \nabla{\phi(0)} + {\ensuremath{\vec{\mathbf{F}}}},

 where :math:`m=m_{i}+m_{s}` is the ice and snow mass per unit area;
:math:`{\ensuremath{\vec{\mathbf{u}}}}=u{\ensuremath{\vec{\mathbf{i}}}}+v{\ensuremath{\vec{\mathbf{j}}}}`
is the ice velocity vector; :math:`{\ensuremath{\vec{\mathbf{i}}}}`,
:math:`{\ensuremath{\vec{\mathbf{j}}}}`, and
:math:`{\ensuremath{\vec{\mathbf{k}}}}` are unit vectors in the
:math:`x`, :math:`y`, and :math:`z` directions, respectively; :math:`f`
is the Coriolis parameter;
:math:`{{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{air}` and
:math:`{{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{ocean}` are the
wind-ice and ocean-ice stresses, respectively; :math:`g` is the gravity
accelation; :math:`\nabla\phi(0)` is the gradient (or tilt) of the sea
surface height; :math:`\phi(0) = g\eta + p_{a}/\rho_{0} + mg/\rho_{0}`
is the sea surface height potential in response to ocean dynamics
(:math:`g\eta`), to atmospheric pressure loading
(:math:`p_{a}/\rho_{0}`, where :math:`\rho_{0}` is a reference density)
and a term due to snow and ice loading :raw-latex:`\citep{cam08}`; and
:math:`{\ensuremath{\vec{\mathbf{F}}}}=\nabla\cdot\sigma` is the
divergence of the internal ice stress tensor :math:`\sigma_{ij}`.
Advection of sea-ice momentum is neglected. The wind and ice-ocean
stress terms are given by

.. math::

   \begin{aligned}
     {{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{air}   = & \rho_{air}  C_{air}   |{\ensuremath{\vec{\mathbf{U}}}}_{air}  -{\ensuremath{\vec{\mathbf{u}}}}|
                      R_{air}  ({\ensuremath{\vec{\mathbf{U}}}}_{air}  -{\ensuremath{\vec{\mathbf{u}}}}), \\ 
     {{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{ocean} = & \rho_{ocean}C_{ocean} |{\ensuremath{\vec{\mathbf{U}}}}_{ocean}-{\ensuremath{\vec{\mathbf{u}}}}| 
                      R_{ocean}({\ensuremath{\vec{\mathbf{U}}}}_{ocean}-{\ensuremath{\vec{\mathbf{u}}}}),\end{aligned}

 where :math:`{\ensuremath{\vec{\mathbf{U}}}}_{air/ocean}` are the
surface winds of the atmosphere and surface currents of the ocean,
respectively; :math:`C_{air/ocean}` are air and ocean drag coefficients;
:math:`\rho_{air/ocean}` are reference densities; and
:math:`R_{air/ocean}` are rotation matrices that act on the wind/current
vectors.

Viscous-Plastic (VP) Rheology[sec:pkg:seaice:VPrheology]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| For an isotropic system the stress tensor :math:`\sigma_{ij}`
  (:math:`i,j=1,2`) can be related to the ice strain rate and strength
  by a nonlinear viscous-plastic (VP) constitutive law
  :raw-latex:`\citep{hib79, zhang97}`:

  .. math::

     \label{eq:vpequation}
       \sigma_{ij}=2\eta(\dot{\epsilon}_{ij},P)\dot{\epsilon}_{ij} 
       + \left[\zeta(\dot{\epsilon}_{ij},P) -
         \eta(\dot{\epsilon}_{ij},P)\right]\dot{\epsilon}_{kk}\delta_{ij}  
       - \frac{P}{2}\delta_{ij}.

   The ice strain rate is given by

  .. math::

     \dot{\epsilon}_{ij} = \frac{1}{2}\left( 
         \frac{\partial{u_{i}}}{\partial{x_{j}}} +
         \frac{\partial{u_{j}}}{\partial{x_{i}}}\right).

   The maximum ice pressure :math:`P_{\max}`, a measure of ice strength,
  depends on both thickness :math:`h` and compactness (concentration)
  :math:`c`:

  .. math::

     P_{\max} = P^{*}c\,h\,\exp\{-C^{*}\cdot(1-c)\},
     \label{eq:icestrength}

   with the constants :math:`P^{*}` (run-time parameter ) and
  :math:`C^{*}=20`. The nonlinear bulk and shear viscosities
  :math:`\eta` and :math:`\zeta` are functions of ice strain rate
  invariants and ice strength such that the principal components of the
  stress lie on an elliptical yield curve with the ratio of major to
  minor axis :math:`e` equal to :math:`2`; they are given by:

  .. math::

     \begin{aligned}
       \zeta =& \min\left(\frac{P_{\max}}{2\max(\Delta,\Delta_{\min})},
        \zeta_{\max}\right) \\
       \eta =& \frac{\zeta}{e^2} \\
       \intertext{with the abbreviation}
       \Delta = & \left[
         \left(\dot{\epsilon}_{11}^2+\dot{\epsilon}_{22}^2\right)
         (1+e^{-2}) +  4e^{-2}\dot{\epsilon}_{12}^2 + 
         2\dot{\epsilon}_{11}\dot{\epsilon}_{22} (1-e^{-2})
       \right]^{\frac{1}{2}}.\end{aligned}

   The bulk viscosities are bounded above by imposing both a minimum
  :math:`\Delta_{\min}` (for numerical reasons, run-time parameter with
  a default value of :math:`10^{-10}\text{\,s}^{-1}`) and a maximum
  :math:`\zeta_{\max} =
  P_{\max}/\Delta^*`, where
  :math:`\Delta^*=(5\times10^{12}/2\times10^4)\text{\,s}^{-1}`. (There
  is also the option of bounding :math:`\zeta` from below by setting
  run-time parameter :math:`>0`, but this is generally not recommended).
  For stress tensor computation the replacement pressure :math:`P
  = 2\,\Delta\zeta` :raw-latex:`\citep{hibler95}` is used so that the
  stress state always lies on the elliptic yield curve by definition.

Defining the CPP-flag in before compiling replaces the method for
bounding :math:`\zeta` by a smooth (differentiable) expression:

.. math::

   \label{eq:zetaregsmooth}
     \begin{split}
     \zeta &= \zeta_{\max}\tanh\left(\frac{P}{2\,\min(\Delta,\Delta_{\min})
         \,\zeta_{\max}}\right)\\
     &= \frac{P}{2\Delta^*}
     \tanh\left(\frac{\Delta^*}{\min(\Delta,\Delta_{\min})}\right) 
     \end{split}

 where :math:`\Delta_{\min}=10^{-20}\text{\,s}^{-1}` is chosen to avoid
divisions by zero.

LSR and JFNK solver [sec:pkg:seaice:LSRJFNK]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  

In the matrix notation, the discretized momentum equations can be
written as

.. math::

   \label{eq:matrixmom}
     {\ensuremath{\mathbf{A}}}({\ensuremath{\vec{\mathbf{x}}}})\,{\ensuremath{\vec{\mathbf{x}}}} = {\ensuremath{\vec{\mathbf{b}}}}({\ensuremath{\vec{\mathbf{x}}}}).

 The solution vector :math:`{\ensuremath{\vec{\mathbf{x}}}}` consists of
the two velocity components :math:`u` and :math:`v` that contain the
velocity variables at all grid points and at one time level. The
standard (and default) method for solving Eq.([eq:matrixmom]) in the sea
ice component of the , as in many sea ice models, is an iterative Picard
solver: in the :math:`k`-th iteration a linearized form
:math:`{\ensuremath{\mathbf{A}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,{\ensuremath{\vec{\mathbf{x}}}}^{k} = {\ensuremath{\vec{\mathbf{b}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})`
is solved (in the case of the MITgcm it is a Line Successive (over)
Relaxation (LSR) algorithm :raw-latex:`\citep{zhang97}`). Picard solvers
converge slowly, but generally the iteration is terminated after only a
few non-linear steps :raw-latex:`\citep{zhang97, lemieux09}` and the
calculation continues with the next time level. This method is the
default method in the MITgcm. The number of non-linear iteration steps
or pseudo-time steps can be controlled by the runtime parameter (default
is 2).

In order to overcome the poor convergence of the Picard-solver,
:raw-latex:`\citet{lemieux10}` introduced a Jacobian-free Newton-Krylov
solver for the sea ice momentum equations. This solver is also
implemented in the MITgcm :raw-latex:`\citep{losch14:_jfnk}`. The Newton
method transforms minimizing the residual
:math:`{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}) = {\ensuremath{\mathbf{A}}}({\ensuremath{\vec{\mathbf{x}}}})\,{\ensuremath{\vec{\mathbf{x}}}} -
{\ensuremath{\vec{\mathbf{b}}}}({\ensuremath{\vec{\mathbf{x}}}})` to
finding the roots of a multivariate Taylor expansion of the residual
:math:`\vec{\mathbf{F}}` around the previous (:math:`k-1`) estimate
:math:`{\ensuremath{\vec{\mathbf{x}}}}^{k-1}`:

.. math::

   \label{eq:jfnktaylor}
     {\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1}+\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}) = 
     {\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1}) + {\ensuremath{\vec{\mathbf{F}}}}'({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}

 with the Jacobian
:math:`{\ensuremath{\mathbf{J}}}\equiv{\ensuremath{\vec{\mathbf{F}}}}'`.
The root
:math:`{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1}+\delta{\ensuremath{\vec{\mathbf{x}}}}^{k})=0`
is found by solving

.. math::

   \label{eq:jfnklin}
     {\ensuremath{\mathbf{J}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,\delta{\ensuremath{\vec{\mathbf{x}}}}^{k} = -{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})

 for :math:`\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}`. The next
(:math:`k`-th) estimate is given by
:math:`{\ensuremath{\vec{\mathbf{x}}}}^{k}={\ensuremath{\vec{\mathbf{x}}}}^{k-1}+a\,\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}`.
In order to avoid overshoots the factor :math:`a` is iteratively reduced
in a line search
(:math:`a=1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \ldots`) until
:math:`\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^k)\| < \|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\|`,
where :math:`\|\cdot\|=\int\cdot\,dx^2` is the :math:`L_2`-norm. In
practice, the line search is stopped at :math:`a=\frac{1}{8}`. The line
search starts after :math:`\code{SEAICE\_JFNK\_lsIter}` non-linear
Newton iterations (off by default).

Forming the Jacobian :math:`{\ensuremath{\mathbf{J}}}` explicitly is
often avoided as “too error prone and time consuming”
:raw-latex:`\citep{knoll04:_jfnk}`. Instead, Krylov methods only require
the action of :math:`\mathbf{J}` on an arbitrary vector
:math:`\vec{\mathbf{w}}` and hence allow a matrix free algorithm for
solving Eq.([eq:jfnklin]) :raw-latex:`\citep{knoll04:_jfnk}`. The action
of :math:`\mathbf{J}` can be approximated by a first-order Taylor series
expansion:

.. math::

   \label{eq:jfnkjacvecfd}
     {\ensuremath{\mathbf{J}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,{\ensuremath{\vec{\mathbf{w}}}} \approx
     \frac{{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1}+\epsilon{\ensuremath{\vec{\mathbf{w}}}}) - {\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})}
     {\epsilon}

 or computed exactly with the help of automatic differentiation (AD)
tools. sets the step size :math:`\epsilon`.

We use the Flexible Generalized Minimum RESidual method
:raw-latex:`\citep[FGMRES,][]{saad93:_fgmres}` with right-hand side
preconditioning to solve Eq.([eq:jfnklin]) iteratively starting from a
first guess of
:math:`\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}_{0} = 0`. For the
preconditioning matrix :math:`\mathbf{P}` we choose a simplified form of
the system matrix
:math:`{\ensuremath{\mathbf{A}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})`
:raw-latex:`\citep{lemieux10}` where
:math:`{\ensuremath{\vec{\mathbf{x}}}}^{k-1}` is the estimate of the
previous Newton step :math:`k-1`. The transformed equation([eq:jfnklin])
becomes

.. math::

   \label{eq:jfnklinpc}
     {\ensuremath{\mathbf{J}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,{\ensuremath{\mathbf{P}}}^{-1}\delta{\ensuremath{\vec{\mathbf{z}}}} =
     -{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1}), 
     \quad\text{with}\quad \delta{\ensuremath{\vec{\mathbf{z}}}}={\ensuremath{\mathbf{P}}}\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}.

 The Krylov method iteratively improves the approximate solution
to ([eq:jfnklinpc]) in subspace
(:math:`{\ensuremath{\vec{\mathbf{r}}}}_0`,
:math:`{\ensuremath{\mathbf{J}}}{\ensuremath{\mathbf{P}}}^{-1}{\ensuremath{\vec{\mathbf{r}}}}_0`,
:math:`({\ensuremath{\mathbf{J}}}{\ensuremath{\mathbf{P}}}^{-1})^2{\ensuremath{\vec{\mathbf{r}}}}_0`,
…,
:math:`({\ensuremath{\mathbf{J}}}{\ensuremath{\mathbf{P}}}^{-1})^m{\ensuremath{\vec{\mathbf{r}}}}_0`)
with increasing :math:`m`;
:math:`{\ensuremath{\vec{\mathbf{r}}}}_0 = -{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})
-{\ensuremath{\mathbf{J}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\,\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}_{0}`
is the initial residual of ([eq:jfnklin]);
:math:`{\ensuremath{\vec{\mathbf{r}}}}_0=-{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})`
with the first guess
:math:`\delta{\ensuremath{\vec{\mathbf{x}}}}^{k}_{0}=0`. We allow a
Krylov-subspace of dimension \ :math:`m=50` and we do not use restarts.
The preconditioning operation involves applying
:math:`{\ensuremath{\mathbf{P}}}^{-1}` to the basis vectors
:math:`{\ensuremath{\vec{\mathbf{v}}}}_0,
{\ensuremath{\vec{\mathbf{v}}}}_1, {\ensuremath{\vec{\mathbf{v}}}}_2, \ldots, {\ensuremath{\vec{\mathbf{v}}}}_m`
of the Krylov subspace. This operation is approximated by solving the
linear system
:math:`{\ensuremath{\mathbf{P}}}\,{\ensuremath{\vec{\mathbf{w}}}}={\ensuremath{\vec{\mathbf{v}}}}_i`.
Because :math:`{\ensuremath{\mathbf{P}}} \approx
{\ensuremath{\mathbf{A}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})`, we
can use the LSR-algorithm :raw-latex:`\citep{zhang97}` already
implemented in the Picard solver. Each preconditioning operation uses a
fixed number of 10 LSR-iterations avoiding any termination criterion.
More details and results can be found in
:raw-latex:`\citet{lemieux10, losch14:_jfnk}`.

To use the JFNK-solver set in the namelist file ; needs to be defined in
and we recommend using a smooth regularization of :math:`\zeta` by
defining (see above) for better convergence. The non-linear Newton
iteration is terminated when the :math:`L_2`-norm of the residual is
reduced by :math:`\gamma_{\mathrm{nl}}` (runtime parameter will already
lead to expensive simulations) with respect to the initial norm:
:math:`\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^k)\| <
\gamma_{\mathrm{nl}}\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^0)\|`.
Within a non-linear iteration, the linear FGMRES solver is terminated
when the residual is smaller than
:math:`\gamma_k\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\|`
where :math:`\gamma_k` is determined by

.. math::

   \label{eq:jfnkgammalin}
     \gamma_k = 
     \begin{cases} 
       \gamma_0 &\text{for $\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\| \geq r$},  \\ 
       \max\left(\gamma_{\min},
       \frac{\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\|}{\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-2})\|}\right)  
       &\text{for $\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{k-1})\| < r$,}
     \end{cases}

 so that the linear tolerance parameter :math:`\gamma_k` decreases with
the non-linear Newton step as the non-linear solution is approached.
This inexact Newton method is generally more robust and computationally
more efficient than exact methods
:raw-latex:`\citep[e.g.,][]{knoll04:_jfnk}`. Typical parameter choices
are :math:`\gamma_0=\code{JFNKgamma\_lin\_max}=0.99`,
:math:`\gamma_{\min}=\code{JFNKgamma\_lin\_min}=0.1`, and :math:`r = 
\code{JFNKres\_tFac}\times\|{\ensuremath{\vec{\mathbf{F}}}}({\ensuremath{\vec{\mathbf{x}}}}^{0})\|`
with :math:`\code{JFNKres\_tFac} = \frac{1}{2}`. We recommend a maximum
number of non-linear iterations :math:`\code{SEAICEnewtonIterMax} = 100`
and a maximum number of Krylov iterations
:math:`\code{SEAICEkrylovIterMax} = 50`, because the Krylov subspace has
a fixed dimension of 50.

Setting turns on “strength implicit coupling”
:raw-latex:`\citep{hutchings04}` in the LSR-solver and in the
LSR-preconditioner for the JFNK-solver. In this mode, the different
contributions of the stress divergence terms are re-ordered in order to
increase the diagonal dominance of the system matrix. Unfortunately, the
convergence rate of the LSR solver is increased only slightly, while the
JFNK-convergence appears to be unaffected.

Elastic-Viscous-Plastic (EVP) Dynamics[sec:pkg:seaice:EVPdynamics]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| :raw-latex:`\citet{hun97}`’s introduced an elastic contribution to the
  strain rate in order to regularize Eq. [eq:vpequation] in such a way
  that the resulting elastic-viscous-plastic (EVP) and VP models are
  identical at steady state,

  .. math::

     \label{eq:evpequation}
       \frac{1}{E}\frac{\partial\sigma_{ij}}{\partial{t}} +
       \frac{1}{2\eta}\sigma_{ij} 
       + \frac{\eta - \zeta}{4\zeta\eta}\sigma_{kk}\delta_{ij}  
       + \frac{P}{4\zeta}\delta_{ij}
       = \dot{\epsilon}_{ij}.

   The EVP-model uses an explicit time stepping scheme with a short
  timestep. According to the recommendation of
  :raw-latex:`\citet{hun97}`, the EVP-model should be stepped forward in
  time 120 times
  (:math:`\code{SEAICE\_deltaTevp} = \code{SEAICIE\_deltaTdyn}/120`)
  within the physical ocean model time step (although this parameter is
  under debate), to allow for elastic waves to disappear. Because the
  scheme does not require a matrix inversion it is fast in spite of the
  small internal timestep and simple to implement on parallel computers
  :raw-latex:`\citep{hun97}`. For completeness, we repeat the equations
  for the components of the stress tensor :math:`\sigma_{1} =
  \sigma_{11}+\sigma_{22}`, :math:`\sigma_{2}= \sigma_{11}-\sigma_{22}`,
  and :math:`\sigma_{12}`. Introducing the divergence :math:`D_D =
  \dot{\epsilon}_{11}+\dot{\epsilon}_{22}`, and the horizontal tension
  and shearing strain rates, :math:`D_T =
  \dot{\epsilon}_{11}-\dot{\epsilon}_{22}` and :math:`D_S =
  2\dot{\epsilon}_{12}`, respectively, and using the above
  abbreviations, the equations [eq:evpequation] can be written as:

  .. math::

     \begin{aligned}
       \label{eq:evpstresstensor1}
       \frac{\partial\sigma_{1}}{\partial{t}} + \frac{\sigma_{1}}{2T} +
       \frac{P}{2T} &= \frac{P}{2T\Delta} D_D \\
       \label{eq:evpstresstensor2}
       \frac{\partial\sigma_{2}}{\partial{t}} + \frac{\sigma_{2} e^{2}}{2T}
       &= \frac{P}{2T\Delta} D_T \\
       \label{eq:evpstresstensor12}
       \frac{\partial\sigma_{12}}{\partial{t}} + \frac{\sigma_{12} e^{2}}{2T}
       &= \frac{P}{4T\Delta} D_S \end{aligned}

   Here, the elastic parameter :math:`E` is redefined in terms of a
  damping timescale :math:`T` for elastic waves

  .. math:: E=\frac{\zeta}{T}.

   :math:`T=E_{0}\Delta{t}` with the tunable parameter :math:`E_0<1` and
  the external (long) timestep :math:`\Delta{t}`.
  :math:`E_{0} = \frac{1}{3}` is the default value in the code and close
  to what :raw-latex:`\citet{hun97}` and :raw-latex:`\citet{hun01}`
  recommend.

To use the EVP solver, make sure that both and are defined in (default).
The solver is turned on by setting the sub-cycling time step to a value
larger than zero. The choice of this time step is under debate.
:raw-latex:`\citet{hun97}` recommend order(120) time steps for the EVP
solver within one model time step :math:`\Delta{t}` (). One can also
choose order(120) time steps within the forcing time scale, but then we
recommend adjusting the damping time scale :math:`T` accordingly, by
setting either (:math:`E_{0}`), so that
:math:`E_{0}\Delta{t}=\mbox{forcing time scale}`, or directly
(:math:`T`) to the forcing time scale.

More stable variants of Elastic-Viscous-Plastic Dynamics: EVP\* , mEVP, and aEVP [sec:pkg:seaice:EVPstar]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| The genuine EVP schemes appears to give noisy solutions
  :raw-latex:`\citep{hun01,
    lemieux12, bouillon13}`. This has lead to a modified EVP or EVP\*
  :raw-latex:`\citep{lemieux12, bouillon13, kimmritz15}`; here, we refer
  to these variants by modified EVP (mEVP) and adaptive EVP (aEVP)
  :raw-latex:`\citep{kimmritz16}`. The main idea is to modify the
  “natural” time-discretization of the momentum equations:

  .. math::

     \label{eq:evpstar}
       m\frac{D\vec{u}}{Dt} \approx m\frac{u^{p+1}-u^{n}}{\Delta{t}}
       + \beta^{*}\frac{u^{p+1}-u^{p}}{\Delta{t}_{\mathrm{EVP}}}

   where :math:`n` is the previous time step index, and :math:`p` is the
  previous sub-cycling index. The extra “intertial” term
  :math:`m\,(u^{p+1}-u^{n})/\Delta{t})` allows the definition of a
  residual :math:`|u^{p+1}-u^{p}|` that, as
  :math:`u^{p+1} \rightarrow u^{n+1}`, converges to :math:`0`. In this
  way EVP can be re-interpreted as a pure iterative solver where the
  sub-cycling has no association with time-relation (through
  :math:`\Delta{t}_{\mathrm{EVP}}`)
  :raw-latex:`\citep{bouillon13, kimmritz15}`. Using the terminology of
  :raw-latex:`\citet{kimmritz15}`, the evolution equations of stress
  :math:`\sigma_{ij}` and momentum :math:`\vec{u}` can be written as:

  .. math::

     \begin{aligned}
       \label{eq:evpstarsigma}
       \sigma_{ij}^{p+1}&=\sigma_{ij}^p+\frac{1}{\alpha}
       \Big(\sigma_{ij}(\vec{u}^p)-\sigma_{ij}^p\Big),
       \phantom{\int}\\
       \label{eq:evpstarmom}
       \vec{u}^{p+1}&=\vec{u}^p+\frac{1}{\beta}
       \Big(\frac{\Delta t}{m}\nabla \cdot{\bf \sigma}^{p+1}+
       \frac{\Delta t}{m}\vec{R}^{p}+\vec{u}_n-\vec{u}^p\Big).\end{aligned}

   :math:`\vec{R}` contains all terms in the momentum equations except
  for the rheology terms and the time derivative; :math:`\alpha` and
  :math:`\beta` are free parameters (, ) that replace the time stepping
  parameters (:math:`\Delta{T}_{\mathrm{EVP}}`), (:math:`E_{0}`), or
  (:math:`T`). :math:`\alpha` and :math:`\beta` determine the speed of
  convergence and the stability. Usually, it makes sense to use
  :math:`\alpha = \beta`, and :math:`\gg
  (\alpha,\,\beta)` :raw-latex:`\citep{kimmritz15}`. Currently, there is
  no termination criterion and the number of mEVP iterations is fixed to
  .

In order to use mEVP in the MITgcm, set in . If the actual form of
equations ([eq:evpstarsigma]) and ([eq:evpstarmom]) is used with fewer
implicit terms and the factor of :math:`e^{2}` dropped in the stress
equations ([eq:evpstresstensor2]) and ([eq:evpstresstensor12]). Although
this modifies the original EVP-equations, it turns out to improve
convergence :raw-latex:`\citep{bouillon13}`.

Another variant is the aEVP scheme :raw-latex:`\citep{kimmritz16}`,
where the value of :math:`\alpha` is set dynamically based on the
stability criterion

.. math::

   \label{eq:aevpalpha}
     \alpha = \beta = \max\left( \tilde{c}\pi\sqrt{c \frac{\zeta}{A_{c}}
       \frac{\Delta{t}}{\max(m,10^{-4}\text{\,kg})}},\alpha_{\min} \right)

 with the grid cell area :math:`A_c` and the ice and snow mass
:math:`m`. This choice sacrifices speed of convergence for stability
with the result that aEVP converges quickly to VP where :math:`\alpha`
can be small and more slowly in areas where the equations are stiff. In
practice, aEVP leads to an overall better convergence than mEVP
:raw-latex:`\citep{kimmritz16}`. To use aEVP in the MITgcm set
:math:`= \tilde{c}`; this also sets the default values of (:math:`c=4`)
and (:math:`\alpha_{\min}=5`). Good convergence has been obtained with
setting these values :raw-latex:`\citep{kimmritz16}`:

Note, that probably because of the C-grid staggering of velocities and
stresses, mEVP may not converge as successfully as in
:raw-latex:`\citet{kimmritz15}`, and that convergence at very high
resolution (order 5km) has not been studied yet.

Truncated ellipse method (TEM) for yield curve [sec:pkg:seaice:TEM]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| In the so-called truncated ellipse method the shear viscosity
  :math:`\eta` is capped to suppress any tensile stress
  :raw-latex:`\citep{hibler97, geiger98}`:

  .. math::

     \label{eq:etatem}
       \eta = \min\left(\frac{\zeta}{e^2},
       \frac{\frac{P}{2}-\zeta(\dot{\epsilon}_{11}+\dot{\epsilon}_{22})}
       {\sqrt{\max(\Delta_{\min}^{2},(\dot{\epsilon}_{11}-\dot{\epsilon}_{22})^2
           +4\dot{\epsilon}_{12}^2})}\right).

   To enable this method, set in and turn it on with in .

Ice-Ocean stress [sec:pkg:seaice:iceoceanstress]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| Moving sea ice exerts a stress on the ocean which is the opposite of
  the stress
  :math:`{{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{ocean}` in
  Eq. [eq:momseaice]. This stess is applied directly to the surface
  layer of the ocean model. An alternative ocean stress formulation is
  given by :raw-latex:`\citet{hibler87}`. Rather than applying
  :math:`{{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{ocean}`
  directly, the stress is derived from integrating over the ice
  thickness to the bottom of the oceanic surface layer. In the resulting
  equation for the *combined* ocean-ice momentum, the interfacial stress
  cancels and the total stress appears as the sum of windstress and
  divergence of internal ice stresses:
  :math:`\delta(z) ({{\ensuremath{\vec{\mathbf{\mathbf{\tau}}}}}}_{air} + {\ensuremath{\vec{\mathbf{F}}}})/\rho_0`,
  :raw-latex:`\citep[see also
  Eq.\,2 of][]{hibler87}`. The disadvantage of this formulation is that
  now the velocity in the surface layer of the ocean that is used to
  advect tracers, is really an average over the ocean surface velocity
  and the ice velocity leading to an inconsistency as the ice
  temperature and salinity are different from the oceanic variables. To
  turn on the stress formulation of :raw-latex:`\citet{hibler87}`, set
  in .

Finite-volume discretization of the stress tensor divergence[sec:pkg:seaice:discretization]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| On an Arakawa C grid, ice thickness and concentration and thus ice
  strength :math:`P` and bulk and shear viscosities :math:`\zeta` and
  :math:`\eta` are naturally defined a C-points in the center of the
  grid cell. Discretization requires only averaging of :math:`\zeta` and
  :math:`\eta` to vorticity or Z-points (or :math:`\zeta`-points, but
  here we use Z in order avoid confusion with the bulk viscosity) at the
  bottom left corner of the cell to give :math:`\overline{\zeta}^{Z}`
  and :math:`\overline{\eta}^{Z}`. In the following, the superscripts
  indicate location at Z or C points, distance across the cell (F),
  along the cell edge (G), between :math:`u`-points (U),
  :math:`v`-points (V), and C-points (C). The control volumes of the
  :math:`u`- and :math:`v`-equations in the grid cell at indices
  :math:`(i,j)` are :math:`A_{i,j}^{w}` and :math:`A_{i,j}^{s}`,
  respectively. With these definitions (which follow the model code
  documentation except that :math:`\zeta`-points have been renamed to
  Z-points), the strain rates are discretized as:

  .. math::

     \begin{aligned}
       \dot{\epsilon}_{11} &= \partial_{1}{u}_{1} + k_{2}u_{2} \\ \notag
       => (\epsilon_{11})_{i,j}^C &= \frac{u_{i+1,j}-u_{i,j}}{\Delta{x}_{i,j}^{F}} 
        + k_{2,i,j}^{C}\frac{v_{i,j+1}+v_{i,j}}{2} \\ 
       \dot{\epsilon}_{22} &= \partial_{2}{u}_{2} + k_{1}u_{1} \\\notag
       => (\epsilon_{22})_{i,j}^C &= \frac{v_{i,j+1}-v_{i,j}}{\Delta{y}_{i,j}^{F}} 
        + k_{1,i,j}^{C}\frac{u_{i+1,j}+u_{i,j}}{2} \\ 
        \dot{\epsilon}_{12} = \dot{\epsilon}_{21} &= \frac{1}{2}\biggl(
        \partial_{1}{u}_{2} + \partial_{2}{u}_{1} - k_{1}u_{2} - k_{2}u_{1}
        \biggr) \\ \notag
       => (\epsilon_{12})_{i,j}^Z &= \frac{1}{2}
       \biggl( \frac{v_{i,j}-v_{i-1,j}}{\Delta{x}_{i,j}^V} 
        + \frac{u_{i,j}-u_{i,j-1}}{\Delta{y}_{i,j}^U} \\\notag
       &\phantom{=\frac{1}{2}\biggl(}
        - k_{1,i,j}^{Z}\frac{v_{i,j}+v_{i-1,j}}{2}
        - k_{2,i,j}^{Z}\frac{u_{i,j}+u_{i,j-1}}{2}
        \biggr),\end{aligned}

   so that the diagonal terms of the strain rate tensor are naturally
  defined at C-points and the symmetric off-diagonal term at Z-points.
  No-slip boundary conditions (:math:`u_{i,j-1}+u_{i,j}=0` and
  :math:`v_{i-1,j}+v_{i,j}=0` across boundaries) are implemented via
  “ghost-points”; for free slip boundary conditions
  :math:`(\epsilon_{12})^Z=0` on boundaries.

For a spherical polar grid, the coefficients of the metric terms are
:math:`k_{1}=0` and :math:`k_{2}=-\tan\phi/a`, with the spherical radius
:math:`a` and the latitude :math:`\phi`;
:math:`\Delta{x}_1 = \Delta{x} = a\cos\phi
\Delta\lambda`, and :math:`\Delta{x}_2 = \Delta{y}=a\Delta\phi`. For a
general orthogonal curvilinear grid, :math:`k_{1}` and :math:`k_{2}` can
be approximated by finite differences of the cell widths:

.. math::

   \begin{aligned}
     k_{1,i,j}^{C} &= \frac{1}{\Delta{y}_{i,j}^{F}}
     \frac{\Delta{y}_{i+1,j}^{G}-\Delta{y}_{i,j}^{G}}{\Delta{x}_{i,j}^{F}} \\
     k_{2,i,j}^{C} &= \frac{1}{\Delta{x}_{i,j}^{F}}
     \frac{\Delta{x}_{i,j+1}^{G}-\Delta{x}_{i,j}^{G}}{\Delta{y}_{i,j}^{F}} \\
     k_{1,i,j}^{Z} &= \frac{1}{\Delta{y}_{i,j}^{U}}
     \frac{\Delta{y}_{i,j}^{C}-\Delta{y}_{i-1,j}^{C}}{\Delta{x}_{i,j}^{V}} \\
     k_{2,i,j}^{Z} &= \frac{1}{\Delta{x}_{i,j}^{V}}
     \frac{\Delta{x}_{i,j}^{C}-\Delta{x}_{i,j-1}^{C}}{\Delta{y}_{i,j}^{U}}\end{aligned}

The stress tensor is given by the constitutive viscous-plastic relation
:math:`\sigma_{\alpha\beta} = 2\eta\dot{\epsilon}_{\alpha\beta} +
[(\zeta-\eta)\dot{\epsilon}_{\gamma\gamma} - P/2
]\delta_{\alpha\beta}` :raw-latex:`\citep{hib79}`. The stress tensor
divergence
:math:`(\nabla\sigma)_{\alpha} = \partial_\beta\sigma_{\beta\alpha}`, is
discretized in finite volumes :raw-latex:`\citep[see
also][]{losch10:_mitsim}`. This conveniently avoids dealing with further
metric terms, as these are “hidden” in the differential cell widths. For
the :math:`u`-equation (:math:`\alpha=1`) we have:

.. math::

   \begin{aligned}
     (\nabla\sigma)_{1}: \phantom{=}&
     \frac{1}{A_{i,j}^w}
     \int_{\mathrm{cell}}(\partial_1\sigma_{11}+\partial_2\sigma_{21})\,dx_1\,dx_2
     \\\notag
     =& \frac{1}{A_{i,j}^w} \biggl\{
     \int_{x_2}^{x_2+\Delta{x}_2}\sigma_{11}dx_2\biggl|_{x_{1}}^{x_{1}+\Delta{x}_{1}}
     + \int_{x_1}^{x_1+\Delta{x}_1}\sigma_{21}dx_1\biggl|_{x_{2}}^{x_{2}+\Delta{x}_{2}}
     \biggr\} \\ \notag
     \approx& \frac{1}{A_{i,j}^w} \biggl\{
     \Delta{x}_2\sigma_{11}\biggl|_{x_{1}}^{x_{1}+\Delta{x}_{1}}
     + \Delta{x}_1\sigma_{21}\biggl|_{x_{2}}^{x_{2}+\Delta{x}_{2}}
     \biggr\} \\ \notag
     =& \frac{1}{A_{i,j}^w} \biggl\{
     (\Delta{x}_2\sigma_{11})_{i,j}^C -
     (\Delta{x}_2\sigma_{11})_{i-1,j}^C 
     \\\notag
     \phantom{=}& \phantom{\frac{1}{A_{i,j}^w} \biggl\{}
     + (\Delta{x}_1\sigma_{21})_{i,j+1}^Z - (\Delta{x}_1\sigma_{21})_{i,j}^Z
     \biggr\}\end{aligned}

 with

.. math::

   \begin{aligned}
     (\Delta{x}_2\sigma_{11})_{i,j}^C =& \phantom{+}
     \Delta{y}_{i,j}^{F}(\zeta + \eta)^{C}_{i,j}
     \frac{u_{i+1,j}-u_{i,j}}{\Delta{x}_{i,j}^{F}} \\ \notag
     &+ \Delta{y}_{i,j}^{F}(\zeta + \eta)^{C}_{i,j}
     k_{2,i,j}^C \frac{v_{i,j+1}+v_{i,j}}{2} \\ \notag
     \phantom{=}& + \Delta{y}_{i,j}^{F}(\zeta - \eta)^{C}_{i,j}
     \frac{v_{i,j+1}-v_{i,j}}{\Delta{y}_{i,j}^{F}} \\ \notag
     \phantom{=}& + \Delta{y}_{i,j}^{F}(\zeta - \eta)^{C}_{i,j}
     k_{1,i,j}^{C}\frac{u_{i+1,j}+u_{i,j}}{2} \\ \notag
     \phantom{=}& - \Delta{y}_{i,j}^{F} \frac{P}{2} \\
     (\Delta{x}_1\sigma_{21})_{i,j}^Z =& \phantom{+}
     \Delta{x}_{i,j}^{V}\overline{\eta}^{Z}_{i,j}
     \frac{u_{i,j}-u_{i,j-1}}{\Delta{y}_{i,j}^{U}} \\ \notag
     & + \Delta{x}_{i,j}^{V}\overline{\eta}^{Z}_{i,j}
     \frac{v_{i,j}-v_{i-1,j}}{\Delta{x}_{i,j}^{V}} \\ \notag
     & - \Delta{x}_{i,j}^{V}\overline{\eta}^{Z}_{i,j} 
     k_{2,i,j}^{Z}\frac{u_{i,j}+u_{i,j-1}}{2} \\ \notag
     & - \Delta{x}_{i,j}^{V}\overline{\eta}^{Z}_{i,j} 
     k_{1,i,j}^{Z}\frac{v_{i,j}+v_{i-1,j}}{2}\end{aligned}

Similarly, we have for the :math:`v`-equation (:math:`\alpha=2`):

.. math::

   \begin{aligned}
     (\nabla\sigma)_{2}: \phantom{=}&
     \frac{1}{A_{i,j}^s}
     \int_{\mathrm{cell}}(\partial_1\sigma_{12}+\partial_2\sigma_{22})\,dx_1\,dx_2 
     \\\notag
     =& \frac{1}{A_{i,j}^s} \biggl\{
     \int_{x_2}^{x_2+\Delta{x}_2}\sigma_{12}dx_2\biggl|_{x_{1}}^{x_{1}+\Delta{x}_{1}}
     + \int_{x_1}^{x_1+\Delta{x}_1}\sigma_{22}dx_1\biggl|_{x_{2}}^{x_{2}+\Delta{x}_{2}}
     \biggr\} \\ \notag
     \approx& \frac{1}{A_{i,j}^s} \biggl\{
     \Delta{x}_2\sigma_{12}\biggl|_{x_{1}}^{x_{1}+\Delta{x}_{1}}
     + \Delta{x}_1\sigma_{22}\biggl|_{x_{2}}^{x_{2}+\Delta{x}_{2}}
     \biggr\} \\ \notag
     =& \frac{1}{A_{i,j}^s} \biggl\{
     (\Delta{x}_2\sigma_{12})_{i+1,j}^Z - (\Delta{x}_2\sigma_{12})_{i,j}^Z
     \\ \notag
     \phantom{=}& \phantom{\frac{1}{A_{i,j}^s} \biggl\{}
     + (\Delta{x}_1\sigma_{22})_{i,j}^C - (\Delta{x}_1\sigma_{22})_{i,j-1}^C
     \biggr\} \end{aligned}

 with

.. math::

   \begin{aligned}
     (\Delta{x}_1\sigma_{12})_{i,j}^Z =& \phantom{+}
     \Delta{y}_{i,j}^{U}\overline{\eta}^{Z}_{i,j}
     \frac{u_{i,j}-u_{i,j-1}}{\Delta{y}_{i,j}^{U}} 
     \\\notag &
     + \Delta{y}_{i,j}^{U}\overline{\eta}^{Z}_{i,j}
     \frac{v_{i,j}-v_{i-1,j}}{\Delta{x}_{i,j}^{V}} \\\notag
     &- \Delta{y}_{i,j}^{U}\overline{\eta}^{Z}_{i,j}
     k_{2,i,j}^{Z}\frac{u_{i,j}+u_{i,j-1}}{2} 
     \\\notag &
     - \Delta{y}_{i,j}^{U}\overline{\eta}^{Z}_{i,j}
     k_{1,i,j}^{Z}\frac{v_{i,j}+v_{i-1,j}}{2} \\ \notag
     (\Delta{x}_2\sigma_{22})_{i,j}^C =& \phantom{+}
     \Delta{x}_{i,j}^{F}(\zeta - \eta)^{C}_{i,j}
     \frac{u_{i+1,j}-u_{i,j}}{\Delta{x}_{i,j}^{F}} \\ \notag
     &+ \Delta{x}_{i,j}^{F}(\zeta - \eta)^{C}_{i,j}
     k_{2,i,j}^{C} \frac{v_{i,j+1}+v_{i,j}}{2} \\ \notag
     & + \Delta{x}_{i,j}^{F}(\zeta + \eta)^{C}_{i,j}
     \frac{v_{i,j+1}-v_{i,j}}{\Delta{y}_{i,j}^{F}} \\ \notag
     & + \Delta{x}_{i,j}^{F}(\zeta + \eta)^{C}_{i,j}
     k_{1,i,j}^{C}\frac{u_{i+1,j}+u_{i,j}}{2} \\ \notag
     & -\Delta{x}_{i,j}^{F} \frac{P}{2}\end{aligned}

Again, no slip boundary conditions are realized via ghost points and
:math:`u_{i,j-1}+u_{i,j}=0` and :math:`v_{i-1,j}+v_{i,j}=0` across
boundaries. For free slip boundary conditions the lateral stress is set
to zeros. In analogy to :math:`(\epsilon_{12})^Z=0` on boundaries, we
set :math:`\sigma_{21}^{Z}=0`, or equivalently :math:`\eta_{i,j}^{Z}=0`,
on boundaries.

Thermodynamics[sec:pkg:seaice:thermodynamics]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| **NOTE: THIS SECTION IS TERRIBLY OUT OF DATE**
| In its original formulation the sea ice model
  :raw-latex:`\citep{menemenlis05}` uses simple thermodynamics following
  the appendix of :raw-latex:`\citet{sem76}`. This formulation does not
  allow storage of heat, that is, the heat capacity of ice is zero.
  Upward conductive heat flux is parameterized assuming a linear
  temperature profile and together with a constant ice conductivity. It
  is expressed as :math:`(K/h)(T_{w}-T_{0})`, where :math:`K` is the ice
  conductivity, :math:`h` the ice thickness, and :math:`T_{w}-T_{0}` the
  difference between water and ice surface temperatures. This type of
  model is often refered to as a “zero-layer” model. The surface heat
  flux is computed in a similar way to that of
  :raw-latex:`\citet{parkinson79}` and :raw-latex:`\citet{manabe79}`.

The conductive heat flux depends strongly on the ice thickness
:math:`h`. However, the ice thickness in the model represents a mean
over a potentially very heterogeneous thickness distribution. In order
to parameterize a sub-grid scale distribution for heat flux
computations, the mean ice thickness :math:`h` is split into :math:`N`
thickness categories :math:`H_{n}` that are equally distributed between
:math:`2h` and a minimum imposed ice thickness of :math:`5\text{\,cm}`
by :math:`H_n= \frac{2n-1}{7}\,h` for :math:`n\in[1,N]`. The heat fluxes
computed for each thickness category is area-averaged to give the total
heat flux :raw-latex:`\citep{hibler84}`. To use this thickness category
parameterization set to the number of desired categories (7 is a good
guess, for anything larger than 7 modify ) in ; note that this requires
different restart files and switching this flag on in the middle of an
integration is not advised. In order to include the same distribution
for snow, set ; only then, the parameterization of always having a
fraction of thin ice is efficient and generally thicker ice is produced
:raw-latex:`\citep{castro-morales14}`.

The atmospheric heat flux is balanced by an oceanic heat flux from
below. The oceanic flux is proportional to
:math:`\rho\,c_{p}\left(T_{w}-T_{fr}\right)` where :math:`\rho` and
:math:`c_{p}` are the density and heat capacity of sea water and
:math:`T_{fr}` is the local freezing point temperature that is a
function of salinity. This flux is not assumed to instantaneously melt
or create ice, but a time scale of three days (run-time parameter ) is
used to relax :math:`T_{w}` to the freezing point. The parameterization
of lateral and vertical growth of sea ice follows that of
:raw-latex:`\citet{hib79, hib80}`; the so-called lead closing parameter
:math:`h_{0}` (run-time parameter ) has a default value of 0.5 meters.

On top of the ice there is a layer of snow that modifies the heat flux
and the albedo :raw-latex:`\citep{zha98a}`. Snow modifies the effective
conductivity according to

.. math:: \frac{K}{h} \rightarrow \frac{1}{\frac{h_{s}}{K_{s}}+\frac{h}{K}},

 where :math:`K_s` is the conductivity of snow and :math:`h_s` the snow
thickness. If enough snow accumulates so that its weight submerges the
ice and the snow is flooded, a simple mass conserving parameterization
of snowice formation (a flood-freeze algorithm following Archimedes’
principle) turns snow into ice until the ice surface is back at
:math:`z=0` :raw-latex:`\citep{leppaeranta83}`. The flood-freeze
algorithm is enabled with the CPP-flag and turned on with run-time
parameter .

Advection of thermodynamic variables[sec:pkg:seaice:advection]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|  
| Effective ice thickness (ice volume per unit area, :math:`c\cdot{h}`),
  concentration :math:`c` and effective snow thickness
  (:math:`c\cdot{h}_{s}`) are advected by ice velocities:

  .. math::

     \label{eq:advection}
       \frac{\partial{X}}{\partial{t}} = - \nabla\cdot\left({\ensuremath{\vec{\mathbf{u}}}}\,X\right) +
       \Gamma_{X} + D_{X}

   where :math:`\Gamma_X` are the thermodynamic source terms and
  :math:`D_{X}` the diffusive terms for quantities
  :math:`X=(c\cdot{h}), c, (c\cdot{h}_{s})`. From the various advection
  scheme that are available in the MITgcm, we recommend flux-limited
  schemes :raw-latex:`\citep[multidimensional 2nd and
  3rd-order advection scheme with flux limiter][]{roe:85, hundsdorfer94}`
  to preserve sharp gradients and edges that are typical of sea ice
  distributions and to rule out unphysical over- and undershoots
  (negative thickness or concentration). These schemes conserve volume
  and horizontal area and are unconditionally stable, so that we can set
  :math:`D_{X}=0`. Run-timeflags: (default=2, is the historic 2nd-order,
  centered difference scheme), = :math:`D_{X}/\Delta{x}`
  (default=0.004).

The MITgcm sea ice model provides the option to use the thermodynamics
model of :raw-latex:`\citet{win00}`, which in turn is based on the
3-layer model of :raw-latex:`\citet{sem76}` and which treats brine
content by means of enthalpy conservation; the corresponding package is
described in section [sec:pkg:thsice]. This scheme requires additional
state variables, namely the enthalpy of the two ice layers (instead of
effective ice salinity), to be advected by ice velocities. The internal
sea ice temperature is inferred from ice enthalpy. To avoid unphysical
(negative) values for ice thickness and concentration, a positive
2nd-order advection scheme with a SuperBee flux limiter
:raw-latex:`\citep{roe:85}` should be used to advect all sea-ice-related
quantities of the :raw-latex:`\citet{win00}` thermodynamic model
(runtime flag and =\ :math:`D_{X}`\ =0 in , defaults are 0). Because of
the non-linearity of the advection scheme, care must be taken in
advecting these quantities: when simply using ice velocity to advect
enthalpy, the total energy (i.e., the volume integral of enthalpy) is
not conserved. Alternatively, one can advect the energy content (i.e.,
product of ice-volume and enthalpy) but then false enthalpy extrema can
occur, which then leads to unrealistic ice temperature. In the currently
implemented solution, the sea-ice mass flux is used to advect the
enthalpy in order to ensure conservation of enthalpy and to prevent
false enthalpy extrema.

Key subroutines [sec:pkg:seaice:subroutines]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Top-level routine:

::


    C     !CALLING SEQUENCE:
    c ...
    c  seaice_model (TOP LEVEL ROUTINE)
    c  |
    c  |-- #ifdef SEAICE_CGRID
    c  |     SEAICE_DYNSOLVER
    c  |     |
    c  |     |-- < compute proxy for geostrophic velocity >
    c  |     |
    c  |     |-- < set up mass per unit area and Coriolis terms >
    c  |     |
    c  |     |-- < dynamic masking of areas with no ice >
    c  |     |
    c  |     |

    c  |   #ELSE
    c  |     DYNSOLVER
    c  |   #ENDIF
    c  |
    c  |-- if ( useOBCS ) 
    c  |     OBCS_APPLY_UVICE
    c  |
    c  |-- if ( SEAICEadvHeff .OR. SEAICEadvArea .OR. SEAICEadvSnow .OR. SEAICEadvSalt )
    c  |     SEAICE_ADVDIFF
    c  |
    c  |-- if ( usePW79thermodynamics ) 
    c  |     SEAICE_GROWTH
    c  |
    c  |-- if ( useOBCS ) 
    c  |     if ( SEAICEadvHeff ) OBCS_APPLY_HEFF
    c  |     if ( SEAICEadvArea ) OBCS_APPLY_AREA
    c  |     if ( SEAICEadvSALT ) OBCS_APPLY_HSALT
    c  |     if ( SEAICEadvSNOW ) OBCS_APPLY_HSNOW
    c  |
    c  |-- < do various exchanges >
    c  |
    c  |-- < do additional diagnostics >
    c  |
    c  o

SEAICE diagnostics [sec:pkg:seaice:diagnostics]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diagnostics output is available via the diagnostics package (see Section
[sec:pkg:diagnostics]). Available output fields are summarized in Table
[tab:pkg:seaice:diagnostics].

Experiments and tutorials that use seaice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Labrador Sea experiment in verification directory.

-  , based on

-  , based on

-  and , global cubed-sphere-experiment with combinations of and
