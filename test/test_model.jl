# mostly checks shapes (model isa Flux.Chain, output size, loss history is finite/non-empty), not that the model is learning correctly.

@testset "build_cnn" begin
    model = build_cnn(imagesize=(64,64,1), num_classes=2)
    @test model isa Flux.Chain

    x = rand(Float32, 64, 64, 1, 4)  # 4 dummy grayscale images
    yn = model(x)
    @test size(yn) == (2, 4)
end

@testset "train_cnn! runs without error" begin
    model = build_cnn(imagesize=(64,64,1), num_classes=2)
    x = rand(Float32, 64, 64, 1, 8)
    y = Flux.onehotbatch(rand(0:1, 8), 0:1)

    loss_history = train_cnn!(model, (x, y), nothing; epochs=1, batchsize=4, device=cpu)

    @test length(loss_history) > 0
    @test all(isfinite, loss_history)
end