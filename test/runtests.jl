using Test, CNNpneumonia
using Flux, Statistics, LinearAlgebra
using CairoMakie

@testset "pneumonia/CNNPneumonia.jl" begin
    include("test_metrics.jl")
    include("test_model.jl")
    include("test_laten_space.jl")
    include("test_plotting.jl")
end