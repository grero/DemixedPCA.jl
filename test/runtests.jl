using DemixedPCA
using Distributions
using Base.Test

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

function test2()
    ncells = 12
    ntrials = 100
    nbins = 100
    t = linspace(0,1,nbins)
    yt = sin.(2*pi*t)
    Xt = reshape(yt,1,1,nbins).*ones(ncells,ntrials,nbins)
    ys = [1.0, 5.0, 10.0]
    label = Array{Int64,1}(ntrials)
    μ = 50*rand(ncells)
    for t in 1:ntrials
        label[t] = rand(1:3)
        for i in 1:ncells
            Xs[i,t,:] = ys[label[t]] + μ[i]
        end
    end
    Fs = randn(ncells,3)
    Ds = randn(3, ncells)
    As = Fs*Ds
    Ft = randn(ncells,1)
    Dt = randn(1,ncells)
    At = Ft*Dt
    X = zeros(ncells, ntrials, nbins)
    for i in 1:nbins
        X[:,:,i] = inv(As)*Xs[:,:,i] + inv(At)*Xt[:,:,i] + 0.1*randn(ncells,ntrials)
    end

    components = DemixedPCA.fit(Demixed.PCA, X, labels)
    components, Fs, Ds, Ft, Dt 
end

