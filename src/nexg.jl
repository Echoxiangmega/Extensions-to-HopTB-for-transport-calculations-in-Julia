# ============================================================
# x=S (spin) or L (orbital)
# Purpose: Calculate intrinsic nonlinear electric-field-induced spin 
#          (and orbital) angular momentum generation.
# Dependencies: HopTB.jl, LinearAlgebra, etc.
# Author: Echo
# Date: 2025-09-04
#
# References:
# - Phys. Rev. Lett. 129, 086602 (2022)
# - npj Spintronics 2, 33 (2024)
#   (Formulas for nonlinear spin and orbital angular momentum generation)
#  Limit: only z-direction orbital angular momentum 
# ============================================================



__precompile__()
module NEXG

using LinearAlgebra, Distributed
using ..HopTB
using ..HopTB.Utilities: fermidirac, constructmeshkpts, splitkpts
using ..HopTB.Parallel: ParallelFunction, claim!, stop!, parallel_sum


export getFS,getiph

const σ1 = [0 1; 1 0]
const σ2 = [0 -im; im 0]
const σ3 = [1 0; 0 -1]
const σs = [σ1, σ2, σ3]
const ita = 0.000
const deg = 0.2e-5
const Gamma=1e-5
const degen_th = [0.2e-5]

function dirac_delta(x; ϵ=5e-6)
    # 这里我们使用一个小的洛伦兹函数来近似狄拉克δ函数
    return ϵ / (π * (x^2 + ϵ^2))
end

function fermi_dirac_derivative(E, E_f, kBT)
    arg = (E - E_f) / kBT
    return -exp(arg) / (kBT * (exp(arg) + 1)^2)
end

function spectral_function(tm::AbstractTBModel, k::AbstractVector{<:Real},bandidx::Integer,E_f)
    egvals, _ = geteig(tm, k)
    En=egvals[bandidx]
    return -1/π*Gamma/((En-E_f)^2+Gamma^2)
end



function remaining_values(n::Int)
    if n == 1
        return (2, 3)
    elseif n == 2
        return (3, 1)
    elseif n == 3
        return (1, 2)
    else
        error("Input should be 1, 2, or 3.")
    end
end

function getspin_pmx(
    tm::AbstractTBModel,
    α::Integer,
    k::AbstractVector{<:Real}
)::Matrix{ComplexF64}
    length(k) == 3 || error("k should be a 3-element vector.")
    α in 1:3 || error("α should be 1, 2 or 3.")
    nspinless = tm.norbits ÷ 2
    V = geteig(tm, k).vectors
    return V' * kron(σs[α], getS(tm, k)[1:nspinless, 1:nspinless]) * V
end

#spin magnetic moment matrix
function getMS(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    s=getspin_pmx(tm,α,k)
    g=2
    #eVT-1
    mub=5.788838e-5  
    return -g*mub*s
end


#spin magnetic moment matrix for a specific band
function getMS_n(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real},bandidx::Integer)::ComplexF64
    s=getspin_pmx(tm,α,k)
    g=2
    #eVT-1
    mub=5.788838e-5
    return -g*mub*s[bandidx,bandidx]
end

#v cross A
function avcross(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    v=HopTB.getvelocity(tm,α,k)
    A=HopTB.getA(tm, β, k)
    egvals, _ = geteig(tm, k)
    mo = zeros(ComplexF64, tm.norbits, tm.norbits)

    for m in 1:tm.norbits, n in 1:tm.norbits
        En=egvals[n]
        for l in 1:tm.norbits
                El=egvals[l]
                if abs(En-El)>deg
                        mo[m, n] +=v[m,l]*A[l,n]/2
                end
                if (l == m)&&(abs(En-El)>deg)
                        mo[m, n] +=v[n,n]*A[l,n]/2
                end
        end
    end
    return mo*1.519267447878626e-05
    #eVT-1
end

function avcross_n(tm::AbstractTBModel, α::Int64, β::Int64, k::AbstractVector{<:Real}, bandidx::Integer)::ComplexF64
    v=HopTB.getvelocity(tm,α,k)
    A=HopTB.getA(tm, β, k)
    egvals = geteig(tm, k).values
    mo = ComplexF64(0.0)

    n = bandidx
  
    En=egvals[n]
    for l in 1:tm.norbits
            El=egvals[l]
            if abs(En-El)>deg
                    mo +=v[n,l]*A[l,n]/2
            end
    end
    return mo*1.519267447878626e-05
end

#orbital magnetic moment matrix
function getMO(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    if α==1
        a=2;b=3
    elseif α==2
        a=3;b=1
    else
        a=1;b=2
    end

    pa=avcross(tm,a,b,k)
    pb=avcross(tm,b,a,k)
    return pa-pb
end

#orbital magnetic moment matrix for a specific band
function getMO_n(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real},bandidx::Integer)::ComplexF64
    if α==1
        a=2;b=3
    elseif α==2
        a=3;b=1
    else
        a=1;b=2
    end

    pa=avcross_n(tm,a,b,k,bandidx)
    pb=avcross_n(tm,b,a,k,bandidx)
    return pa-pb
end

#orbital angular momentum matrix z-direction lack minus sign
function getoamz(tm::TBModel,k::AbstractVector{<:Real})
  va=HopTB.getvelocity(tm,1,k);vb=HopTB.getvelocity(tm,2,k)
  egvals, _ = geteig(tm, k)
  ag = zeros(ComplexF64, tm.norbits, tm.norbits)
  for n in 1:tm.norbits
      En=egvals[n]
      for m in 1:tm.norbits
         Em=egvals[m]
         #fm = fermidirac(0.0, Em-μ)
         #if n==m
         #   tmp=1
         #else
         #   tmp=abs(fm-fn)
         #end
         for l in 1:tm.norbits
                El=egvals[l]
            if (l==n) || (l==m)
              continue
            end
            if abs(El-En) > degen_th[1]
              eln=1/(El-En)
            else
              eln=0
            end
            if abs(El-Em) > degen_th[1]
              elm=1/(El-Em)
            else
              elm=0
            end
            ag[n,m] -= im*(va[n,l]*vb[l,m]-vb[n,l]*va[l,m])*(eln+elm)*0.0656171059018794
         end
      end
  end
  return ag
end




#fermi surface orbital
function getint1(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,bandidx::Integer)
  Lz=getoamz(tm,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=real(Lz[n,n]*va[n,m]*vb[m,n]/(Es[n]-Es[m]+im*ita)^3)
        result-=real(Lz[n,m]*(va[m,n]*vb[n,n]+vb[m,n]*va[n,n])/(Es[n]-Es[m]+im*ita)^3)
        result-=real(Lz[n,m]*(va[m,m]*vb[m,n]+vb[m,m]*va[m,n])/(Es[n]-Es[m]+im*ita)^3/2)
      end

    end
  end
  return -1*result
end

#fermi surface spin
function getint1s(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer)
  Lz=getspin_pmx(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=real(Lz[n,n]*va[n,m]*vb[m,n]/(Es[n]-Es[m]+im*ita)^3)
        result-=real(Lz[n,m]*(va[m,n]*vb[n,n]+vb[m,n]*va[n,n])/(Es[n]-Es[m]+im*ita)^3)
        result-=real(Lz[n,m]*(va[m,m]*vb[m,n]+vb[m,m]*va[m,n])/(Es[n]-Es[m]+im*ita)^3/2)
      end

    end
  end
  return -1*result
end

#fermi surface orbital worker
function int1_worker(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs))
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, _ = geteig(tm, k)
        for n in db:ub
            ϵn = egvals[n]
            theta1=getint1(tm,k,a, b,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs) 
                  if abs(ϵn-μ)<0.15              
                  	fn = fermi_dirac_derivative(ϵn,μ,T)
                  	result[iT,iμ]+=fn*theta1
	          end
            end
        end
    end
    return result
end

#fermi sea orbital
function getint2(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,bandidx::Integer)
  Lz=getoamz(tm,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=3*real((Lz[n,n]-Lz[m,m])*va[n,m]*vb[m,n]/(Es[n]-Es[m]+im*ita)^4)
        for l in 1:tm.norbits
          if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*Lz[n,l]/(Es[n]-Es[m]+im*ita)^3/(Es[n]-Es[l]+im*ita))
          end
          if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*Lz[m,l]/(Es[n]-Es[m]+im*ita)^3/(Es[m]-Es[l]+im*ita))
          end
        end
      end

    end
  end
  return -1*result
end

#fermi sea spin
function getint2s(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer)
  Lz=getspin_pmx(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=3*real((Lz[n,n]-Lz[m,m])*va[n,m]*vb[m,n]/(Es[n]-Es[m]+im*ita)^4)
        for l in 1:tm.norbits
          if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*Lz[n,l]/(Es[n]-Es[m]+im*ita)^3/(Es[n]-Es[l]+im*ita))
          end
          if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*Lz[m,l]/(Es[n]-Es[m]+im*ita)^3/(Es[m]-Es[l]+im*ita))
          end
        end
      end

    end
  end
  return -1*result
end

#fermi sea orbital worker
function int2_worker(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs))
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, _ = geteig(tm, k)
        for n in db:ub
            ϵn = egvals[n]
            theta1=getint2(tm,k,a, b,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)               
                  fn = fermidirac(T/8.617333262145e-5, ϵn-μ)
                  result[iT,iμ]+=fn*theta1
            end
        end
    end
    return result
end

#all term spin
function ints_worker(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64,c::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs))
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, _ = geteig(tm, k)
        for n in db:ub
            ϵn = egvals[n]
            theta1=getint2s(tm,k,a, b,c,n)
            theta2=getint1s(tm,k,a, b,c,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)
                  fn = fermidirac(T/8.617333262145e-5, ϵn-μ)
                  result[iT,iμ]+=fn*theta1
		          if abs(ϵn-μ)<0.15
                        fm = fermi_dirac_derivative(ϵn,μ,T)
                        result[iT,iμ]+=fm*theta2
                  end
            end
        end
    end
    return result
end





#orbital-related fermi sea term
function getintsea(tm::AbstractTBModel, a::Int64, b::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(int2_worker, tm, a, b,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

#orbital-related fermi surface term
function getintsurface(tm::AbstractTBModel, a::Int64, b::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(int1_worker, tm, a, b,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

#all spin-related terms
function getints(tm::AbstractTBModel, a::Int64, b::Int64,c::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ints_worker, tm, a, b,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

end
