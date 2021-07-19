using DemixedPCA
using Distributions
using LinearAlgebra
using Random
using Test

function test1()
    RNG = MersenneTwister(UInt32(1234))
	X,labels = DemixedPCA.generate_data(RNG)
	ppc = StatsBase.fit(DemixedPCA.dPCA, X, labels)
	y = DemixedPCA.transform(ppc, X)
	y
end

#y = test1()
#@show sum(y)
#@test sum(y) ≈ -18624.818676126706

function test2(RNG=MersenneTwister(rand(UInt32)))
    ncells = 12
    ntrials = 100
    nbins = 100
    t = range(0,stop=1,length=nbins)
    #a single time component
    yt = sin.(2*pi*t)
    Xt = reshape(yt,1,1,nbins).*ones(1,ntrials,nbins)
    Xs = fill(0.0, 3, ntrials, nbins)
    ys = [1.0, 5.0, 10.0]
    label = Vector{Int64}(undef, ntrials)
    μ = 50*rand(RNG, ncells)
    for t in 1:ntrials
        label[t] = rand(RNG, 1:3)
        for i in 1:3
            Xs[i,t,:] .= ys[label[t]]
        end
    end
    Fs = randn(RNG, ncells,3)
    Ds = randn(RNG, 3, ncells)
    As = Fs*Ds
    Ft = randn(RNG, ncells,1)
    @show Ft
    Dt = randn(RNG, 1,ncells)
    At = Ft*Dt
    X = fill(0.0, ncells, ntrials, nbins)
    for i in 1:nbins
        X[:,:,i] .= Fs*Xs[:,:,i] + Ft*Xt[:,:,i] + 0.1*randn(ncells,ntrials)
    end

    components = DemixedPCA.fit(DemixedPCA.dPCA, X, label)
    Zt = DemixedPCA.transform(components["time"], X)
    @show size(Zt), size(Xt)
    @show norm(Zt .- Xt)
    components, Fs, Ds, Ft, Dt
end

components, Fs, Ds, Ft, Dt = test2()
@show Ft
@show components["time"].F
@show components["time"].D


