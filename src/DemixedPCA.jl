module DemixedPCA
using MultivariateStats
import MultivariateStats.transform
import StatsBase.fit

type dPCA
  F::Array{Float64,2}
  D::Array{Float64,2}
	pca::PCA
	a_ols::Array{Float64,2}
end

"""
Fit a demixed-PCA model to the data in `X`. We have `(d,n1,n2) = size(X)`, where `d` is the dimension and `n1` and `n2` are observations along different dimensions, e.g. realizations and time, respectively.

Kobak, D., Brendel, W., Constantinidis, C., & Feierstein, C. E. (2016). Demixed principal component analysis of neural population data. eLife. http://doi.org/10.7554/eLife.10989.001
"""
function fit(::Type{dPCA},X::Array{Float64,3},labels::Array{Int64,1};maxoutdim=3,λ=0.0)::dPCA
	ncells,ntrials,nbins = size(X)
	#get the average firing rate for each cell
	μ = mean(X, (2,3))
  Xϕ = zeros(X)
  ulabels = unique(labels)
  sort!(ulabels)
  #marginalize over the labels
  for (i,l) in enumerate(ulabels)
    _idx = labels .== l
    for jj in 1:size(X,1)
			for kk in 1:size(X,3)
				_m = mean(X[jj,_idx,kk]- μ[jj])
				Xϕ[jj,_idx,kk] .= _m
			end
    end
  end
	Y = reshape(permutedims(X .-μ, [3,2,1]), ntrials*nbins, ncells)'
	Yϕ = reshape(permutedims(Xϕ, [3,2,1]), ntrials*nbins, ncells)'
	#marginalize over time

  #solve the ordinary least square problem, i.e. argmin ||Xϕ - XA_OLS||
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
  return dPCA(F, D,mmc,A_OLS)
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

