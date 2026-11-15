include("CNNpneumonia.jl")
using .CNNpneumonia, CairoMakie
using Flux: cpu, onecold, softmax
using BSON: @load
using Flux

const MODEL_PATH = joinpath(@__DIR__, "..", "models", "pneumonia", "model_frozen_90.5.bson")
@load MODEL_PATH model

x_train, y_train, x_val, y_val, x_test, y_test = get_data("pneumoniamnist")

samples = 200
x_batch = x_test[:, :, :, 1:samples]   
labels = onecold(y_test[:, 1:samples], label_names)#[1]

activations = extract_latent_space(model, x_batch)
coords = pca_2d(activations)

plot_latent_space(coords, labels; filename="latent_space_pca.png")

Flux.testmode!(model)
logits = model(x_batch)
y_hat = label_names[onecold(logits)]
y = label_names[onecold(y_test[:, 1:samples])]

plot_latent_space_errors(coords, y_hat, y)
