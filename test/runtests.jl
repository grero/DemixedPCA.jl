using DemixedPCA
using Distributions
using Base.Test

function test1()
	srand(1234)
	X,labels = DemixedPCA.generate_data()
	ppc = StatsBase.fit(DemixedPCA.dPCA, X, labels)
	y = DemixedPCA.transform(ppc, X)
	y
end

y = test1()
@test_approx_eq sum(y) -18624.818676126706


