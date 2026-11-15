# checks that a Figure object comes back and a file gets written, not that the plot looks right.

@testset "plot_latent_space" begin
    coords = randn(Float32, 2, 20)
    labels = rand(["normal", "pneumonia"], 20)

    fig = CNNpneumonia.plot_latent_space(coords, labels)
    @test fig isa Figure

    mktempdir() do dir
        path = joinpath(dir, "latent.png")
        CNNpneumonia.plot_latent_space(coords, labels; filename=path)
        @test isfile(path)
    end
end

@testset "plot_training_loss_history" begin
    mktempdir() do dir
        plot_training_loss_history(Float32[1.0, 0.8, 0.5], "loss.png", dir)
        @test isfile(joinpath(dir, "loss.png"))
    end
end