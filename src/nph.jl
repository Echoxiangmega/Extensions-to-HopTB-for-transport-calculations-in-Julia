__precompile__()
module NPH

using LinearAlgebra, Distributed,Statistics
using DelimitedFiles
using ..HopTB
using ..HopTB.KCache32Module
using ..HopTB.Utilities: fermidirac, constructmeshkpts, splitkpts
using ..HopTB.Parallel: ParallelFunction, claim!, stop!, parallel_sum


export getFS,getiph

const σ1 = [0 1; 1 0]
const σ2 = [0 -im; im 0]
const σ3 = [1 0; 0 -1]
const σs = [σ1, σ2, σ3]
const ita = 0
const deg = 0.2e-5

function dirac_delta(x; ϵ=1e-5)
    # 这里我们使用一个小的洛伦兹函数来近似狄拉克δ函数
    return ϵ / (π * (x^2 + ϵ^2))
end

function fermi_dirac_derivative(E, E_f, kBT)
    arg = (E - E_f) / kBT
    return -exp(arg) / (kBT * (exp(arg) + 1)^2)
end


function fermi_dirac_derivative!(res::Float64, E, E_f, kBT)
    res = 0.0
    arg = (E - E_f) / kBT
    res -= exp(arg) / (kBT * (exp(arg) + 1)^2)
    return nothing
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

function getMS(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real})::Matrix{ComplexF64}
    s=getspin_pmx(tm,α,k)
    g=2
    #eVT-1
    mub=5.788838e-5  
    return -g*mub*s
end

function getMS_n(tm::AbstractTBModel, α::Int64, k::AbstractVector{<:Real},bandidx::Integer)::ComplexF64
    s=getspin_pmx(tm,α,k)
    g=2
    #eVT-1
    mub=5.788838e-5
    return -g*mub*s[bandidx,bandidx]
end


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


function getlam(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer)
  ms=getMS(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=3*real(va[n,m]*vb[m,n]*(ms[n,n]-ms[m,m])/(Es[n]-Es[m]+im*ita)^4)
        for l in 1:tm.norbits
	  if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*ms[n,l]/(Es[n]-Es[m]+im*ita)^3/(Es[n]-Es[l]+im*ita))
          end
	  if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*ms[m,l]/(Es[n]-Es[m]+im*ita)^3/(Es[m]-Es[l]+im*ita))
          end
        end    
      end
      
    end
  end
  return 2*result
end

function getlam_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer)
  ms=getMO(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=3*real(va[n,m]*vb[m,n]*(ms[n,n]-ms[m,m])/(Es[n]-Es[m]+im*ita)^4)
        for l in 1:tm.norbits
          if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*ms[n,l]/(Es[n]-Es[m]+im*ita)^3/(Es[n]-Es[l]+im*ita))
          end
          if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*ms[m,l]/(Es[n]-Es[m]+im*ita)^3/(Es[m]-Es[l]+im*ita))
          end
        end
      end

    end
  end
  return 2*result
end

function getlam2omega(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer,omega)
  ms=getMS(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      #if abs(Es[n]-Es[m])>deg && abs(omega-abs(Es[n]-Es[m]))>deg && !(m in degenerate_band_indices)
      if !(m in degenerate_band_indices)
        result+=real(va[n,m]*vb[m,n]*(ms[n,n]-ms[m,m])*(3*(Es[n]-Es[m])^2-omega^2)/(Es[n]-Es[m]+im*ita)^2/(omega^2-(Es[n]-Es[m]+im*ita)^2)^2)
        for l in 1:tm.norbits
	  if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*ms[n,l]/(Es[m]-Es[n]+im*ita)/(omega^2-(Es[n]-Es[m]+im*ita)^2)/(Es[n]-Es[l]))
          end
	  if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*ms[m,l]/(Es[m]-Es[n]+im*ita)/(omega^2-(Es[n]-Es[m]+im*ita)^2)/(Es[m]-Es[l]))
          end
        end    
      end
      
    end
  end
  return 2*result
end


function getlam2omega_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer,omega)
  ms=getMO(tm,c,k)
  va=HopTB.getvelocity(tm,a, k)
  vb=HopTB.getvelocity(tm,b, k)

  Es=geteig(tm, k).values
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      #if abs(Es[n]-Es[m])>deg && abs(omega-abs(Es[n]-Es[m]))>deg && !(m in degenerate_band_indices)
      if !(m in degenerate_band_indices)
        result+=real(va[n,m]*vb[m,n]*(ms[n,n]-ms[m,m])*(3*(Es[n]-Es[m])^2-omega^2)/(Es[n]-Es[m]+im*ita)^2/(omega^2-(Es[n]-Es[m]+im*ita)^2)^2)
        for l in 1:tm.norbits
          if abs(Es[n]-Es[l])>deg
             result-=real((va[l,m]*vb[m,n]+vb[l,m]*va[m,n])*ms[n,l]/(Es[m]-Es[n]+im*ita)/(omega^2-(Es[n]-Es[m]+im*ita)^2)/(Es[n]-Es[l]))
          end
          if abs(Es[m]-Es[l])>deg
             result-=real((va[l,n]*vb[n,m]+vb[l,n]*va[n,m])*ms[m,l]/(Es[m]-Es[n]+im*ita)/(omega^2-(Es[n]-Es[m]+im*ita)^2)/(Es[m]-Es[l]))
          end
        end
      end

    end
  end
  return 2*result
end


function getdG(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer)
  Aa=HopTB.getA(tm,a, k)
  Ab=HopTB.getA(tm,b, k)

  dAa=getdr(tm,a,c,k)
  dAb=getdr(tm,b,c,k)

  Es=geteig(tm, k).values
  dEs = getdEs(tm, c, k)
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg && !(m in degenerate_band_indices)
        result+=2*real(dAa[n,m]*Ab[m,n]/(Es[n]-Es[m]+im*ita))
        result+=2*real(Aa[n,m]*dAb[m,n]/(Es[n]-Es[m]+im*ita))
        result-=2*real(Aa[n,m]*Ab[m,n]*(dEs[n]-dEs[m])/(Es[n]-Es[m]+im*ita)^2)
      end
    end
  end
  return result
end

function getdG2omega(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,bandidx::Integer,omega)
  Aa=HopTB.getA(tm,a, k)
  Ab=HopTB.getA(tm,b, k)

  dAa=getdr(tm,a,c,k)
  dAb=getdr(tm,b,c,k)

  Es=geteig(tm, k).values
  dEs = getdEs(tm, c, k)
  degenerate_band_indices=[bandidx]

  result=0.0
  for n in degenerate_band_indices
    for m in 1:tm.norbits

      if abs(Es[n]-Es[m])>deg  && !(m in degenerate_band_indices)
        result+=2*real(dAa[n,m]*Ab[m,n]*(Es[m]-Es[n])/(omega^2-(Es[n]-Es[m]+im*ita)^2))
        result+=2*real(Aa[n,m]*dAb[m,n]*(Es[m]-Es[n])/(omega^2-(Es[n]-Es[m]+im*ita)^2))
        result-=2*real(Aa[n,m]*Ab[m,n]*(dEs[n]-dEs[m])*(omega^2+(Es[n]-Es[m]+im*ita)^2)/(omega^2-(Es[n]-Es[m]+im*ita)^2)^2)
      end
    end
  end
  return result
end


function getalpha(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k);vb=getdEs(tm,b,k)
  lam1=getlam(tm,k,b,c,d,bandidx);lam2=getlam(tm,k,a,c,d,bandidx)
  dG1=getdG(tm,k,b,c,a,bandidx);dG2=getdG(tm,k,a,c,b,bandidx)
  ms=getMS_n(tm,d,k,bandidx)

  result=0.0

  result+=real(va[bandidx]*lam1-vb[bandidx]*lam2)
  result+=real((dG1-dG2)*ms)

  
  return result
end


function getalpha1(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k);vb=getdEs(tm,b,k)
  lam1=getlam(tm,k,b,c,d,bandidx);lam2=getlam(tm,k,a,c,d,bandidx)


  result=0.0

  result+=real(va[bandidx]*lam1-vb[bandidx]*lam2)

  
  return result
end

function getalpha2(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  dG1=getdG(tm,k,b,c,a,bandidx);dG2=getdG(tm,k,a,c,b,bandidx)
  ms=getMS_n(tm,d,k,bandidx)

  result=0.0

  result+=real((dG1-dG2)*ms)

  
  return result
end

function getalpha1_disp(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k)
  lam1=getlam(tm,k,b,c,d,bandidx)


  result=0.0

  result=0.5*real(va[bandidx]*lam1)

  
  return result
end

function getalpha2_disp(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  dG1=getdG(tm,k,b,c,a,bandidx)
  ms=getMS_n(tm,d,k,bandidx)

  result=0.0

  result=0.5*real((dG1)*ms)

  
  return result
end

function getalpha1_disp_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k)
  lam1=getlam_o(tm,k,b,c,d,bandidx)


  result=0.0

  result=0.5*real(va[bandidx]*lam1)


  return result
end

function getalpha2_disp_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  dG1=getdG(tm,k,b,c,a,bandidx)
  ms=getMO_n(tm,d,k,bandidx)

  result=0.0

  result=0.5*real((dG1)*ms)


  return result
end


function getalpha12omega(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer,omega)
  va=getdEs(tm,a,k)
  lam1=getlam2omega(tm,k,b,c,d,bandidx,omega)


  result=0.0

  result-=0.5*real(va[bandidx]*lam1)

  
  return result
end

function getalpha22omega(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer,omega)
  dG1=getdG2omega(tm,k,b,c,a,bandidx,omega)
  ms=getMS_n(tm,d,k,bandidx)

  result=0.0

  result-=0.5*real((dG1)*ms)

  
  return result
end

function getalpha12omega_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer,omega)
  va=getdEs(tm,a,k)
  lam1=getlam2omega_o(tm,k,b,c,d,bandidx,omega)


  result=0.0

  result-=0.5*real(va[bandidx]*lam1)


  return result
end

function getalpha22omega_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer,omega)
  dG1=getdG2omega(tm,k,b,c,a,bandidx,omega)
  ms=getMO_n(tm,d,k,bandidx)

  result=0.0

  result-=0.5*real((dG1)*ms)


  return result
end

function getalpha_o(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k);vb=getdEs(tm,b,k)
  lam1=getlam_o(tm,k,b,c,d,bandidx);lam2=getlam_o(tm,k,a,c,d,bandidx)
  dG1=getdG(tm,k,b,c,a,bandidx);dG2=getdG(tm,k,a,c,b,bandidx)
  ms=getMO_n(tm,d,k,bandidx)

  result=0.0

  result+=real(va[bandidx]*lam1-vb[bandidx]*lam2)
  result+=real((dG1-dG2)*ms)


  return result
end

function getalpha_o1(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  va=getdEs(tm,a,k);vb=getdEs(tm,b,k)
  lam1=getlam_o(tm,k,b,c,d,bandidx);lam2=getlam_o(tm,k,a,c,d,bandidx)

  result=0.0

  result+=real(va[bandidx]*lam1-vb[bandidx]*lam2)



  return result
end

function getalpha_o2(tm::AbstractTBModel,k::AbstractVector{<:Real},a::Int64, b::Int64,c::Int64,d::Int64,bandidx::Integer)
  dG1=getdG(tm,k,b,c,a,bandidx);dG2=getdG(tm,k,a,c,b,bandidx)
  ms=getMO_n(tm,d,k,bandidx)

  result=0.0
  result+=real((dG1-dG2)*ms)


  return result
end


function ipho_worker(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 19:20
            ϵn = egvals[n]
            theta1=getalpha1(tm,k,a, b, c,d,n)
            theta2=getalpha2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.000173)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end

function ipho_workerf(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha1(tm,k,a, b, c,d,n)
            theta2=getalpha2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.0004325)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end

function ipho_workerf_disp(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha1_disp(tm,k,a, b, c,d,n)
            theta2=getalpha2_disp(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,T)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_workerf_disp_o(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha1_disp_o(tm,k,a, b, c,d,n)
            theta2=getalpha2_disp_o(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,T)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_worker9(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 17:20
            ϵn = egvals[n]
            theta1=getalpha1(tm,k,a, b, c,d,n)
            theta2=getalpha2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.0004325)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_worker_mo(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs))
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 13:14
            ϵn = egvals[n]
            theta=getalpha(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.12
                  fn=fermi_dirac_derivative(ϵn,μ,0.00173)
                  result[iT,iμ]+=fn*theta
               end


            end
        end
    end
    return result
end


function ipho_workero(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs))
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 15:20
            ϵn = egvals[n]
            theta=getalpha_o(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.12
                  fn=fermi_dirac_derivative(ϵn,μ,0.00347)
                  result[iT,iμ]+=fn*theta
               end


            end
        end
    end
    return result
end


function ipho_workero_fen(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 19:20
            ϵn = egvals[n]
            theta1=getalpha_o1(tm,k,a, b, c,d,n)
            theta2=getalpha_o2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.000173)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_workero_fenfkk(
    kpts::Matrix{Float64}, tm::AbstractTBModel, 
    a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64, ub::Int64,omega::Float64 ,
    Ts::Vector{Float64}, μs::Vector{Float64}
)
    nkpts = size(kpts, 2)
    μnum = length(μs)
    contribs = zeros(μnum, nkpts)
    klist = Vector{NTuple{3, Float64}}(undef, nkpts)

    for ik in 1:nkpts
        k = kpts[:, ik]
        kval = (
            round(Float64(k[1]), digits=8),
            round(Float64(k[2]), digits=8),
            round(Float64(k[3]), digits=8)
        )
        klist[ik] = kval
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha12omega_o(tm,k,a, b, c,d,n,omega)
            theta2=getalpha22omega_o(tm,k,a, b, c,d,n,omega)
            for (iμ, μ) in enumerate(μs)
                if abs(ϵn - μ) < 0.025
                    f1 = fermi_dirac_derivative(ϵn, μ, Ts[1])
                    contribs[iμ, ik] += real(f1 * (theta1+theta2))
                end
            end
        end
    end

    return klist, contribs
end



function findkimport(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    nkpts = size(kpts, 2)
    result = zeros(length(μs),nkpts)
    max_values=zeros(length(μs),nkpts)
    re = []

    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha_o1(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.0004325)
                  result[iμ,ik]+=fn*theta1
                  max_values[iμ,ik] = abs(result[iμ,ik])
               end


            end
        end
    end

    for (iμ, μ) in enumerate(μs)
              threshold=quantile(max_values[iμ,:],0.98)
              significant_kpts = findall(max_values[iμ,:] .> threshold)
              if threshold>1e10
		re=union(re,significant_kpts)
              end

    end


    return kpts[:,re]
end

function ipho_workero_fenf(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    nkpts = size(kpts, 2)
    result = zeros(length(Ts),length(μs),2)
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha_o1(tm,k,a, b, c,d,n)
            theta2=getalpha_o2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.0004325)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end

function ipho_workero_fenf2omega(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    omega::Float64,db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    nkpts = size(kpts, 2)
    result = zeros(length(Ts),length(μs),2)
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha12omega(tm,k,a, b, c,d,n,omega)
            theta2=getalpha22omega(tm,k,a, b, c,d,n,omega)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.25
                  fn=fermi_dirac_derivative(ϵn,μ,T)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_workero_fenf2omega_o(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    omega::Float64,db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    nkpts = size(kpts, 2)
    result = zeros(length(Ts),length(μs),2)
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha12omega_o(tm,k,a, b, c,d,n,omega)
            theta2=getalpha22omega_o(tm,k,a, b, c,d,n,omega)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.25
                  fn=fermi_dirac_derivative(ϵn,μ,T)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end


function ipho_workero_fen9(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64})
    result = zeros(length(Ts),length(μs),2)
    nkpts = size(kpts, 2)
    theta = 0.0
    fn = 0.0
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals = geteig(tm, k).values
        for n in 17:20
            ϵn = egvals[n]
            theta1=getalpha_o1(tm,k,a, b, c,d,n)
            theta2=getalpha_o2(tm,k,a, b, c,d,n)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)

               if abs(ϵn-μ)<0.025
                  fn=fermi_dirac_derivative(ϵn,μ,0.0004325)
                  result[iT,iμ,1]+=fn*theta1
                  result[iT,iμ,2]+=fn*theta2
               end


            end
        end
    end
    return result
end

function getnph(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_worker9, tm, a, b, c,d, Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_o(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh), nworkers())
    pf = ParallelFunction(ipho_workero, tm, a, b, c,d, Ts, μs)

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

function getnph_o_fen(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workero_fen9, tm, a, b, c,d, Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_o_fen2(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workero_fenf, tm, a, b, c,d,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_o_fen22omega(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,omega::Float64,
    nkmesh::Vector{Int64};Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workero_fenf2omega, tm, a, b, c,d,omega,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_o_fen22omega_o(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,omega::Float64,
    nkmesh::Vector{Int64};Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workero_fenf2omega_o, tm, a, b, c,d,omega,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function extract_kpoint_weights(kvec::Vector{Float64}, m::Int, L::Int)
    @assert length(kvec) == 3 + m * L
    raw_weights = kvec[4:end]
    return reshape(raw_weights, m, L)'  # 输出为 L × m第一个指标是mu,第二个是能带
end

function ipho_workero_fenf2omega_gailv(kpts::Matrix{Float64}, tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    omega::Float64,db::Int64,ub::Int64,Ts::Vector{Float64}, μs::Vector{Float64})
    nkpts = size(kpts, 2)
    result = zeros(length(Ts),length(μs))
    fn = 0.0
    L=length(μs)
    m=ub-db+1
    for ik in 1:nkpts
        kvec = kpts[:, ik]
        k=kvec[1:3]
        w = extract_kpoint_weights(kvec, m, L) #L*m size
        egvals = geteig(tm, k).values
        for n in db:ub
            ϵn = egvals[n]
            theta1=getalpha12omega(tm,k,a, b, c,d,n,omega)
            theta2=getalpha22omega(tm,k,a, b, c,d,n,omega)
            for (iT, T) in enumerate(Ts),(iμ, μ) in enumerate(μs)
                  result[iT,iμ,1]+=w[iμ,n-db+1]*(theta1+theta2)


            end
        end
    end
    return result
end


function getnphkk(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workero_fenfkk, tm, a, b, c,d,db,ub,Ts, μs)
    
    results=nothing
    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    #results=[]
    for iw in 1:nworkers()
        worker_result = claim!(pf)
        filename = "worker_$(iw)_results.dat"
        filename1 = "worker_$(iw)_kmesh.dat"
            open(filename, "w") do f
                writedlm(f, worker_result)
            end
	    open(filename1, "w") do f
                writedlm(f, kptslist[iw])
            end
    end
    stop!(pf)

    #return results,tmp
end


function getnph2(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workerf, tm, a, b, c,d,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph2_disp(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workerf_disp, tm, a, b, c,d,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph2_disp_o(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ipho_workerf_disp_o, tm, a, b, c,d,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_ada(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0],iter::Int64=1)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(ada_ipho_workero_fenf, tm, a, b, c,d,db,ub,Ts, μs,iter)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts),iter+1, length(μs),2)
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_findk(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],offset::Vector=[0.0,0.0,0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0],iter::Int64=1)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh;offset=offset,k1=k1,k2=k2), nworkers())
    pf = ParallelFunction(findkimport, tm, a, b, c,d,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    re = zeros(3,1)
    for iw in 1:nworkers()
        tmp=claim!(pf)
        if tmp!=[]
            re=hcat(re,tmp)
        end
    end
    stop!(pf)

    return re # e**2/(hbar*(2π)^3)*1.0e10/100
end

function getnph_o_fen22omega_adaptive(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64, ub::Int64, omega::Float64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],
    offset::Vector{Float64}=[0.0, 0.0, 0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0],
    refine_ratio::Float64=0.9, nrefine::Vector{Int}=[3, 3, 3], niter::Int=1)

    Δk = [(k2[1]-k1[1]) / nkmesh[1], (k2[2]-k1[2]) / nkmesh[2], (k2[3]-k1[3]) / nkmesh[3]]
    bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2]) ⋅ tm.rlat[:, 3])
    cache = Dict{NTuple{3, Float64}, Vector{Float64}}()

    kpts = eachcol(constructmeshkpts(nkmesh; offset=offset, k1=k1, k2=k2))
    w_base = 1.0 / length(kpts)
    tol = 1e-4  # 可调，积分误差容忍度
    last_result = nothing
    max_refine_per_mu = 20

    
    for iter in 1:niter
        println("Iteration $iter")
        shrink = 0.98*0.455^(iter-1)
        new_kpts_list = Vector{Vector{Float64}}()

	    println(size(kpts))
        for k in kpts
            @assert length(k) == 3 "Bad input k: $(k)"
            kvec = (
                round(Float64(k[1]), digits=8),
                round(Float64(k[2]), digits=8),
                round(Float64(k[3]), digits=8)
            )
            push!(new_kpts_list, collect(kvec))
            #if (iter == 1 && !haskey(cache, kvec)) || 
            #    (iter > 1 && any(isnan, cache[kvec][1:μnum]))
            #     push!(new_kpts_list, collect(kvec))
            #end             
        end

        if !isempty(new_kpts_list)
            new_kpts_mat = hcat(new_kpts_list...)
            chunks = splitkpts(new_kpts_mat, nworkers())
            pf = ParallelFunction(ipho_workero_fenfkk, tm, a, b, c, d, db, ub, omega, Ts, μs)
            for chunk in chunks
                pf(chunk)
            end
            for _ in 1:length(chunks)
                klist, values = claim!(pf)
                for j in 1:length(klist)
                    kvec = klist[j]
                    v = values[:, j]
                    weight = (iter == 1) ? w_base : cache[kvec][end]
                    cache[kvec] = vcat(v, [weight])
                end
            end
            stop!(pf)
        end

        # 简单积分估计
        μnum = length(μs)
        current_result = zeros(length(Ts), μnum)

        valid = 0
        nonzero = 0

        for v in values(cache)
            w = v[end]
            if w === missing || isnan(v[1])
                continue
            end
            valid += 1
            if any(x -> abs(x) > 1e-12, v[1:μnum])
                nonzero += 1
            end
            for iμ in 1:μnum
                current_result[1, iμ] += v[iμ] * w
            end
        end

        current_result .*= bzvol / (2π)^3
        println("Valid points in cache: $valid, with nonzero value: $nonzero")
        #println("iter = $current_result")

        if iter < niter
            μnum = length(μs)
            refine_set = Set{NTuple{3, Float64}}()
            for iμ in 1:μnum
                values_μ = [abs(v[iμ]) for v in values(cache) if v[end] !== missing && !isnan(v[iμ])]
                cutoff = quantile(values_μ, refine_ratio)
                

                candidates_μ = [
                    kkey for (kkey, v) in cache
                    if v[end] !== missing && !isnan(v[iμ]) && abs(v[iμ]) > cutoff && abs(v[iμ]) > 1e-4
                ]

                # 降序按贡献值排序，只取前 max_refine_points 个
                sorted_candidates = sort(candidates_μ, by = kkey -> -abs(cache[kkey][iμ]))
                selected_μ = first(sorted_candidates, min(max_refine_per_mu, length(sorted_candidates)))
                #for (kkey, v) in cache
                #    if v[end] !== missing && !isnan(v[iμ]) && abs(v[iμ]) > cutoff
                #        push!(refine_set, kkey)
                #    end
                #end
                # Step 5: 合并进总 refine_set（自动去重）
                union!(refine_set, selected_μ)
            end

            for kkey in refine_set
                kcenter = collect(kkey)
                v_parent = cache[kkey]
                w_parent = v_parent[end]
                delete!(cache, kkey)
                w_refined = w_parent / prod(nrefine)
                

                for i = 0:nrefine[1]-1, j = 0:nrefine[2]-1, l = 0:nrefine[3]-1
                    δx = (nrefine[1] == 1) ? 0.0 : (i - (nrefine[1]-1)/2) * Δk[1] * shrink / (nrefine[1]-1)
                    δy = (nrefine[2] == 1) ? 0.0 : (j - (nrefine[2]-1)/2) * Δk[2] * shrink / (nrefine[2]-1)
                    δz = (nrefine[3] == 1) ? 0.0 : (l - (nrefine[3]-1)/2) * Δk[3] * shrink / (nrefine[3]-1)
                    kp = kcenter .+ [δx, δy, δz]
                    kvec = Tuple(round.(kp, digits=8))
                    
                    
                    if !haskey(cache, kvec)
                        cache[kvec] = vcat(fill(NaN, length(μs)), [w_refined])
                    end
                end
                
            end


            kpts = [collect(k) for (k, v) in cache if isnan(v[1]) && v[end] !== missing]
        end
        

        # 如果不是第一轮，比较与上轮差异
        if last_result !== nothing
            diff = maximum(abs.(current_result .- last_result))
            println("Iteration $iter: max result diff = $diff")
            if diff < tol
                println("Refine terminated early due to convergence.")
                break
            end
        end

        # 更新用于下轮比较
        last_result = copy(current_result)

    end

    uncomputed = [collect(k) for (k, v) in cache if isnan(v[1]) && v[end] !== missing]
    if !isempty(uncomputed)
        new_kpts_mat = hcat(uncomputed...)
        chunks = splitkpts(new_kpts_mat, nworkers())
        pf = ParallelFunction(ipho_workero_fenfkk, tm, a, b, c, d, db, ub, omega, Ts, μs)
        for chunk in chunks
            pf(chunk)
        end
        for _ in 1:length(chunks)
            klist, values = claim!(pf)
            for j in 1:length(klist)
                kvec = klist[j]
                v = values[:, j]
                weight = cache[kvec][end]
                cache[kvec] = vcat(v, [weight])
            end
        end
        stop!(pf)
    end

    μnum = length(μs)
    total_result = zeros(length(Ts), μnum)
    for v in values(cache)
        w = v[end]
        if w === missing || isnan(v[1])
            continue
        end
        for iμ in 1:μnum
            total_result[1, iμ] += v[iμ] * w
        end
    end

    return total_result * bzvol / (2π)^3
end

function getnph_o_fen22omega_adaptive2(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64, d::Int64,
    db::Int64, ub::Int64, omega::Float64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0],
    offset::Vector{Float64}=[0.0, 0.0, 0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0],
    refine_ratio::Float64=0.9, nrefine::Vector{Int}=[3, 3, 3], niter::Int=1)

    Δk = [(k2[1]-k1[1]) / nkmesh[1], (k2[2]-k1[2]) / nkmesh[2], (k2[3]-k1[3]) / nkmesh[3]]
    bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2]) ⋅ tm.rlat[:, 3])
    kcache = init_kcache32()

    kpts = eachcol(constructmeshkpts(nkmesh; offset=offset, k1=k1, k2=k2))
    w_base = 1.0 / length(kpts)
    μm = length(μs)
    tol = 1e-3
    last_result = nothing
    max_refine_per_mu = 20
    m= 15

    for iter in 1:niter
        println("Iteration $iter")
        shrink = 0.5 * 0.25^(iter - 1)

        new_kpts_list = NTuple{3, Float32}[]
        for k in kpts
            kvec = Tuple(round.(Float32.(k), digits=6))
            if iter == 1 || (haskey(kcache.kmap, kvec) && isnan(get_kcache(kcache, kvec)[1]))
                push!(new_kpts_list, kvec)
            end
        end

        if !isempty(new_kpts_list)
            new_kpts_mat = hcat([Float64.(collect(k)) for k in new_kpts_list]...)
            chunks = splitkpts(new_kpts_mat, nworkers())
            pf = ParallelFunction(ipho_workero_fenfkk, tm, a, b, c, d, db, ub, omega, Ts, μs)

            for chunk in chunks
                pf(chunk)
            end
            for _ in 1:length(chunks)
                klist, values = claim!(pf)
                for j in 1:length(klist)
                    kvec = Tuple(round.(Float32.(klist[j]), digits=6))
                    v = Float32.(values[:, j])
                    weight = (iter == 1) ? Float32(w_base) : get_kcache(kcache, kvec)[end]
                    update_kcache!(kcache, kvec, vcat(v, [weight]))
                end
            end
            stop!(pf)
        end

        result = zeros(Float64, length(Ts), μm)
        valid = 0
        nonzero = 0

        for v in kcache.vlist
            w = v[end]
            if isnan(v[1])
                continue
            end
            valid += 1
            if any(abs.(v[1:μm]) .> 1e-12)
                nonzero += 1
            end
            for iμ in 1:μm
                result[1, iμ] += v[iμ] * w
            end
        end

        result .*= bzvol / (2π)^3
        println("Valid k-points: $valid, Nonzero: $nonzero")

        save_kcache_hdf5("refine_iter_$iter.h5", kcache, 1e-1)

        if iter < niter
            refine_set = Set{NTuple{3, Float32}}()
	    diff_refine_set = Set{NTuple{3, Float32}}()
            for iμ in 1:μm
                values_μ = [abs(v[iμ]) for v in kcache.vlist if !isnan(v[iμ])]
                if isempty(values_μ)
                    continue
                end
                cutoff = quantile(values_μ, refine_ratio)
                candidates = [k for (k, idx) in kcache.kmap if !isnan(kcache.vlist[idx][iμ]) && abs(kcache.vlist[idx][iμ]) > cutoff && kcache.vlist[idx][end] > w_base/(prod(nrefine))^2]
                sorted = sort(candidates, by=k -> -abs(kcache.vlist[kcache.kmap[k]][iμ]))
                selected = first(sorted, min(max_refine_per_mu, length(sorted)))
                union!(refine_set, selected)
            end

            for iμ in 2:μm
   		 scored = []

		 for (k, idx) in kcache.kmap
       			v = kcache.vlist[idx]
        		if v[end] === missing || isnan(v[iμ]) || isnan(v[iμ - 1])
        		    continue
        		end
        		score = abs(v[iμ] - v[iμ - 1])
        		push!(scored, (score, k))
    		 end

  	         # 取 top m 个差值大的 k 点
    		 sorted = sort(scored, by = x -> -x[1])
    		 topk = first(sorted, min(length(sorted), m))

    		 for (_, k) in topk
        		push!(diff_refine_set, k)
    		 end
	    end
            
	    union!(refine_set,diff_refine_set)


            for kvec in refine_set
                kcenter = collect(Float64.(kvec))
                v_parent = get_kcache(kcache, kvec)
                w_parent = v_parent[end]
                w_refined = w_parent / prod(nrefine)
                delete!(kcache.kmap, kvec)

                for i in 0:nrefine[1]-1, j in 0:nrefine[2]-1, l in 0:nrefine[3]-1
                    δx = (nrefine[1] == 1) ? 0.0 : (i - (nrefine[1]-1)/2) * Δk[1] * shrink / (nrefine[1]-1)
                    δy = (nrefine[2] == 1) ? 0.0 : (j - (nrefine[2]-1)/2) * Δk[2] * shrink / (nrefine[2]-1)
                    δz = (nrefine[3] == 1) ? 0.0 : (l - (nrefine[3]-1)/2) * Δk[3] * shrink / (nrefine[3]-1)
                    kp = Tuple(round.(Float32.(kcenter .+ [δx, δy, δz]), digits=6))
                    if !haskey(kcache.kmap, kp)
                        update_kcache!(kcache, kp, vcat(fill(Float32(NaN), μm), [Float32(w_refined)]))
                    end
                end
            end

            kpts = uncomputed_kpoints(kcache)
        end

        if last_result !== nothing
            diff = maximum(abs.(result .- last_result))
            println("Max diff: $diff")
            if diff < tol
                println("Converged.")
                break
            end
        end

        last_result = copy(result)
    end

    total_result = zeros(Float64, length(Ts), μm)
    for v in kcache.vlist
        w = v[end]
        if isnan(v[1])
            continue
        end
        for iμ in 1:μm
            total_result[1, iμ] += v[iμ] * w
        end
    end

    return total_result * bzvol / (2π)^3
end


function getnph_o_fen22omega_gailv(tm::AbstractTBModel, a::Int64, b::Int64, c::Int64,d::Int64,db::Int64,ub::Int64,
    omega::Float64,kvec::AbstractMatrix{Float64};Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0])
    kptslist = splitkpts(kvec, nworkers())
    pf = ParallelFunction(ipho_workero_fenf2omega_gailv, tm, a, b, c,d,omega,db,ub,Ts, μs)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/(2π)^3 # e**2/(hbar*(2π)^3)*1.0e10/100
end

end
