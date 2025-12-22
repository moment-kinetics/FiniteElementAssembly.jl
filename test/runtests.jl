module FiniteElementAssemblyTests

using Test

@testset "FiniteElementAssemblyTests" begin
    include(joinpath(@__DIR__, "CalculusTests.jl"))
    include(joinpath(@__DIR__, "InterpolationTests.jl"))
    include(joinpath(@__DIR__, "InterfaceTests.jl"))
    include(joinpath(@__DIR__, "PoissonSolverZedRadialTests.jl"))
    include(joinpath(@__DIR__, "PoissonSolverThetaZedTests.jl"))
    include(joinpath(@__DIR__, "PoissonSolverRadialThetaTests.jl"))
    include(joinpath(@__DIR__, "PeriodicBcTests.jl"))
    include(joinpath(@__DIR__, "ElectronIonCollisionsTest.jl"))
    include(joinpath(@__DIR__, "TestParticleCollisionsVpaVperp.jl"))
    include(joinpath(@__DIR__, "SlowingDownTest.jl"))
end

end # FiniteElementAssemblyTests