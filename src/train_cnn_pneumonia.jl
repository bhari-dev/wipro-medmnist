include("CNNpneumonia.jl")
using .CNNpneumonia, CairoMakie
using Flux: cpu, gpu, onecold, softmax
using Flux.Optimisers: Adam, Descent
using LinearAlgebra: tr
using BSON: @save


x_train, y_train, x_val, y_val, x_test, y_test = get_data("pneumoniamnist")

device = cpu
println("buiding model...")
model = build_cnn(; imagesize=size(x_train)[1:end-1], num_classes=size(y_train, 1)) |> device
class_counts = vec(sum(y_train, dims=2))

inverse_freq_weights = sum(class_counts) ./ (length(class_counts) .* class_counts)
class_weights = Float32.(sqrt.(inverse_freq_weights)) |> device

println("starting training....")
loss_history = train_cnn!(model, (x_train, y_train), (x_test, y_test); epochs=1, weights=class_weights);
println("finished training")
threshold = 0.93
probs = softmax(model(x_test |> cpu) |> cpu)  # (2, N)
predictions = [probs[2, i] > threshold ? "pneumonia" : "normal" for i in axes(probs, 2)]

confMat = confusion_matrix(predictions, onecold(y_test, label_names))
plot_confusion_matrix(confMat)

println("\nPer-class accuracy:")
per_class_accuracy(confMat)

plot_training_loss_history(loss_history)



accuracy = tr(confMat) / sum(confMat)
println("Accuracy: ", accuracy)


save_dir = joinpath(@__DIR__, "..", "models", "pneumonia")
mkpath(save_dir)
@save joinpath(save_dir, "model_frozen_$(round(accuracy*100, digits=2))%_accuracy.bson") model