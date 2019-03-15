module DemixedPCA
using MultivariateStats
import MultivariateStats.transform
import StatsBase.fit
include("utils.jl")

export dPCA

struct dPCA
    F::Array{Float64,2}
    D::Array{Float64,2}
    pca::PCA
	a_ols::Array{Float64,2}
end

"""
Perform dPCA by finding an optimal set of weights that minimize the difference betwen Yϕ and FDY.
"""
function compute_dpca(Y::Matrix{Float64}, Yϕ::Matrix{Float64};maxoutdim=3, λ=0.0)
  if λ == 0.0
    A_OLS = Yϕ/Y
  else
    A_OLS = ridge(Y, Yϕ, λ)
  end
  _Y = A_OLS*Y
  #get the low rank approximation using PCA
  mmc = fit(PCA, _Y;maxoutdim=maxoutdim)
  pp = projection(mmc)
  #Aq=pp*pp'*A_OLS
  F = pp
  D = pp'*A_OLS
  dPCA(F,D, mmc, A_OLS)
end

function marginalize(X::Array{Float64,3}, trial_labels::AbstractArray{Int64,1})
    ncells, ntrials, nbins = size(X)
    labels = unique(trial_labels)
    sort!(labels)
    Xp = fill!(similar(X), 0.0)
    for ll in labels
        tidx = trial_labels.==ll
        μt = mean(X[:,tidx,:], dims=(2,3)).*ones(ncells,sum(tidx),nbins)
        Xp[:,tidx,:] = μt 
    end
    Xp
end

"""
Fit a demixed-PCA model to the data in `X`. We have `(d,n1,n2) = size(X)`, where `d` is the dimension and `n1` and `n2` are observations along different dimensions, e.g. realizations (trials) and time, respectively.

Kobak, D., Brendel, W., Constantinidis, C., & Feierstein, C. E. (2016). Demixed principal component analysis of neural population data. eLife. http://doi.org/10.7554/eLife.10989.001
"""
function fit(::Type{dPCA},X::Array{Float64,3},labels::Array{Int64,1};maxoutdim=3,λ=0.0)
	ncells,ntrials,nbins = size(X)
	#get the average firing rate for each cell
	μ = mean(X, dims=(2,3))
    
    components = Dict()
    Y = permutedims(reshape(permutedims(X .- μ, [3,2,1]), ntrials*nbins, ncells),[2,1])
    #time component
    Yt = permutedims(reshape(permutedims(mean(X .-μ, dims=2).*ones(size(X)), [3,2,1]), ntrials*nbins, ncells),[2,1])
    components["time"] = compute_dpca(Y,Yt,maxoutdim=maxoutdim)
    #stimulus component
    Xs = marginalize(X .- μ, labels)
    Ys = permutedims(reshape(permutedims(Xs, [3,2,1]), ntrials*nbins, ncells),[2,1])
    components["stimulus"] = compute_dpca(Y, Ys,maxoutdim=maxoutdim)
    return components
end

"""
Transform X by projecting it onto the demixed PCA space `ppc`. Note that `size(X) = (ndims, ntrials, nbins)`
"""
function transform(ppc::dPCA, X::Array{Float64,3})
	y = zeros(size(ppc.D,1), size(X,2), size(X,3))
	for i in 1:size(y,3)
		y[:,:,i] = ppc.D*X[:,:,i]
	end
	y
end

end #module

