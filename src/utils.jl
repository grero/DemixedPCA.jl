using Distributions
"""
Create a model where the firing rates are given as 
"""
function generate_data(RNG=MersenneTwister(rand(UInt32)))
	λ = [1.0, 5.0,10.0,15, 25]
	nstim = length(λ)	
	ncells = 10
	nbins = 10	
	ntrials = 100
	X = zeros(ncells, nbins,ntrials)
	labels = zeros(Int64,ntrials)
	label_idx = zeros(Int64,nstim,ncells)
	for i in 1:ncells
		label_idx[:,i] = shuffle(RNG, 1:nstim)
	end
	for i in 1:ntrials
		s = rand(RNG, 1:nstim)
		labels[i] = s
		for j in 1:nbins
			for c in 1:ncells
				X[c, j,i] = rand(RNG, Poisson(λ[label_idx[s,c]]))
			end
		end
	end
	permutedims(X, [1,3,2]), labels
end
