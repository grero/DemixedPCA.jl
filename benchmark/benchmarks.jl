using PkgBenchmark
using DemixedPCA

X,labels = DemixedPCA.generate_data()
@benchgroup "default" begin
	@bench "fit" ppc = StatsBase.fit(DemixedPCA.dPCA, X, labels)
end
