# Extensions-to-HopTB-for-transport-calculations-in-Julia

This repository contains Julia extensions to [HopTB](https://github.com/HopTB/HopTB.jl) for transport and response calculations.

## New Module 1: `src/nexg.jl`

The file [`src/nexg.jl`](src/nexg.jl) implements routines to compute the **intrinsic contribution** of **spin or orbital angular momentum polarization** induced by a nonlinear electric field.  

This extension allows you to evaluate how electric fields can drive angular momentum generation beyond linear response, with direct connection to recent theoretical developments in condensed matter physics.

### Usage
To use this module, copy `src/nexg.jl` into the `src/` folder of your local HopTB installation and include it in the HopTB.jl:

```julia
#submodules
include("nexg.jl")
```

### References

The formulas implemented in `nexg.jl` are based on the following works:

- [Phys. Rev. Lett. **129**, 086602 (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.086602)  
- [*npj Spintronics* **2**, 33 (2024)](https://www.nature.com/articles/s44306-024-00041-4)


### Usage Example

You can compute the intrinsic nonlinear electric-field-induced spin polarization using:

```julia
HopTB.NEXG.getints(tb_model_from_wannier, alpha, beta, c, 1, ub, kmesh; Ts=[0.00086173], μs=ωs)
```

In this command:
- `alpha, beta, c` are tensor component indices, each taking values in {1, 2, 3}.
- `ub` is the highest band index used in the calculation.
- `kmesh` sets the density of the k-point mesh.
- `μs` is an array of Fermi energy values.
- `Ts` is the temperature array, here given in energy units (eV, with 0.00086173 ≈ 10 K).

## New Module 2: `src/nph.jl`

The file [`src/nph.jl`](src/nph.jl) implements routines to compute the **intrinsic nonlinear planar Hall effect (NPH)**.

### Usage
To use this module, copy `src/nph.jl` into the `src/` folder of your local HopTB installation and include it in the HopTB.jl:

```julia
#submodules
include("nph.jl")

### Usage Example

You can compute the intrinsic nonlinear planar Hall response (orbital part) using:

```julia
HopTB.NPH.getnph_o_fen2(tb_model_from_wannier, alpha, beta, gamma, d, db, ub, kmesh; μs=ωs)
```
In this command:
- `alpha, beta, gamma, d` are tensor component indices, each taking values in {1, 2, 3}.
- `db, ub` indicate the band index range used in the calculation.
- `kmesh` sets the density of the k-point mesh.
- `μs` is an array of Fermi energy values.

### References

The formulas implemented in `nph.jl` are based on the following works:

- [PhysRevLett.130.126303](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.126303)  

