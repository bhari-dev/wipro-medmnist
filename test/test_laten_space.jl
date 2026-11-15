# checks output shape and that PC1 has more variance than PC2 on synthetic data engineered to have that property.

@testset "pca_2d" begin
    activations = zeros(Float32, 3, 100)
    activations[1, :] = 10f0 .* randn(Float32, 100)   # high variance
    activations[2, :] = 0.1f0 .* randn(Float32, 100)  # low variance
    activations[3, :] = 0.1f0 .* randn(Float32, 100)

    coords = CNNpneumonia.pca_2d(activations)

    @test size(coords) == (2, 100)
    @test var(coords[1, :]) > var(coords[2, :])  # PC1 should capture more variance
end

@testset "extract_latent_space" begin
    model = build_cnn(imagesize=(64,64,1), num_classes=2)
    x = rand(Float32, 64, 64, 1, 6)

    acts = CNNpneumonia.extract_latent_space(model, x)

    @test size(acts, 2) == 6          # one column per sample
end