module CNNpneumonia

using Flux, Flux.Losses, CUDA, 
		MLDatasets, PlutoUI, CairoMakie, 
		LinearAlgebra, FileIO, OneHotArrays,
		Downloads, NPZ, Colors, ImageBase
	
	CairoMakie.activate!(; px_per_unit = 4)

using Statistics: mean
using LinearAlgebra: svd

export get_data, build_cnn, train_cnn!, label_names,
	confusion_matrix, per_class_accuracy, plot_training_loss_history,
	plot_confusion_matrix, extract_latent_space, pca_2d, plot_latent_space, plot_latent_space_errors,
	saliency_map, smoothgrad, arr_to_img, alpha_cmap



const DATA_FOLDER = joinpath(@__DIR__, "..","data")

"""
    get_data(dataset_name, resolution=64)

Load a MedMNIST-style dataset from `data/<dataset_name/<dataset_name>_<resolution>.npz`
and prepare it for training with Flux.

Images are transposed into Flux's `(H, W, C, N)` convention and scaled to `[0, 1]`.
Labels are one-hot encoded over the sorted unique label values found in the training split.

# Arguments
- `dataset_name::AbstractString`: name of the dataset (case-insensitive), e.g. `"pneumoniamnist"`.
  Must match a subfolder under `data/` and the corresponding `.npz` file name.
- `resolution::Integer=64`: image resolution to load (must match an available `.npz` file,
  e.g. `data/pneumoniamnist/pneumoniamnist_64.npz`).

# Returns
A 6-element `Vector` of preprocessed splits, in order:
`[train_images, train_labels, val_images, val_labels, test_images, test_labels]`.
Images are `Float32` arrays of size `(H, W, C, N)`; labels are one-hot matrices.

# Notes
If the expected `.npz` file is missing, an `@info` message is logged pointing you to the
data-download script (see the README) before attempting to read the file, which will error.
"""
function get_data(dataset_name, resolution=64)
	dataset_name = lowercase(dataset_name)
	data_path = joinpath(DATA_FOLDER, dataset_name, "$(dataset_name)_$(resolution).npz")
	if !isfile(data_path)
		@info "The file $data_path does not exists, please use the script to download the data"
	end
	data = npzread(data_path)
	labels = minimum(data["train_labels"]):maximum(data["train_labels"])
	# grayscale: (N,H,W) → (H,W,1,N); RGB would be (N,H,W,C) → (H,W,C,N)
	img_preprocess_fn = img -> begin
		p = permutedims(img, (2:ndims(img)..., 1))
		ndims(p) == 3 ? reshape(p, size(p, 1), size(p, 2), 1, size(p, 3)) : p
	end
	lbl_preprocess_fn = lbl -> onehotbatch(lbl[:, 1], labels)
	# pre-processing
	for data_type in ("train", "val", "test")
		data["$(data_type)_images"] = img_preprocess_fn(data["$(data_type)_images"])
		data["$(data_type)_labels"] = lbl_preprocess_fn(data["$(data_type)_labels"])
	end
	return [
		data["train_images"] ./ 255f0, data["train_labels"],
		data["val_images"] ./ 255f0, data["val_labels"],
		data["test_images"] ./ 255f0, data["test_labels"]
	]
end

"""
    label_names

Ordered class names for the classification task, indexed the same way as one-hot
label rows and confusion-matrix rows/columns: `["normal", "pneumonia"]`.
"""
label_names = ["normal", "pneumonia"]

"""
    build_cnn(; imagesize, num_classes)

Construct the CNN architecture used for pneumonia classification: three
`Conv -> BatchNorm -> MaxPool` blocks (16 -> 32 -> 64 channels) followed by two
dense layers.

# Arguments
- `imagesize::Tuple`: input image size as `(H, W, C)`, e.g. `(64, 64, 1)` for
  64x64 grayscale images. Note: the flatten -> dense transition is hard-coded for a
  64x64 input (`8*8*64 = 4096` features after three 2x2 max-pools); other sizes will
  require updating the first `Dense` layer's input width.
- `num_classes::Integer`: number of output classes (e.g. `2` for normal/pneumonia).

# Returns
A `Flux.Chain` representing the untrained model.

# Example
```julia
model = build_cnn(imagesize=(64, 64, 1), num_classes=2)
```
"""
function build_cnn(; imagesize, num_classes)
	cnn = Chain(
		Conv((3, 3), 1 => 16, relu, pad=1),
		BatchNorm(16, relu),
		MaxPool((2,2)),
		
		Conv((3,3), 16 => 32, relu, pad=1),
		BatchNorm(32, relu),
		MaxPool((2,2)),

		Conv((3,3), 32 => 64, relu, pad=1),
		BatchNorm(64, relu),
		MaxPool((2,2)),

		Flux.flatten,

		# 8x8x64 = 4096

		Dense(4096 => 128, relu),
		Dense(128 => num_classes),

	)

	return cnn
end

"""
    train_cnn!(model, train_data, test_data; epochs=15, batchsize=128, device=cpu,
               optimizer=Adam(1e-3), weights=ones(Float32, length(label_names)),
               loss_function=logitcrossentropy)

Train `model` in place on `train_data` using gradient descent, printing the
latest batch loss at the end of every epoch.

# Arguments
- `model`: a Flux model (e.g. from [`build_cnn`](@ref)), updated in place.
- `train_data`: a `(images, labels)` tuple passed to `Flux.DataLoader`.
- `test_data`: currently unused by the training loop itself (accepted for API
  compatibility / future evaluation hooks); pass `nothing` if you don't need it.

# Keyword Arguments
- `epochs::Integer=15`: number of passes over `train_data`.
- `batchsize::Integer=128`: mini-batch size.
- `device`: `cpu` or `gpu` (from Flux), applied to each batch and to `weights` before training.
- `optimizer`: a Flux/Optimisers.jl optimizer rule, default `Adam(1e-3)`.
- `weights`: per-class weights used to weight the loss (see `label_names` for class order).
- `loss_function`: loss function applied to `(logits, y; agg)`, default `logitcrossentropy`.

# Returns
A `Vector` of the per-batch training loss recorded across all epochs (in order).

# Example
```julia
model = build_cnn(imagesize=(64, 64, 1), num_classes=2)
loss_history = train_cnn!(model, (x_train, y_train), nothing; epochs=10)
```
"""
function train_cnn!(model, train_data, test_data; 
	epochs=15, batchsize=128, 
	device=cpu, 
	optimizer=Adam(1e-3), 
	weights=ones(Float32, length(label_names)) |> device,
	loss_function=logitcrossentropy)

	#setup data and model
	train_loader = Flux.DataLoader(train_data, batchsize=batchsize, shuffle=true)
	optimizer_state = Flux.setup(optimizer, model)

	#perform training
	loss_history = []
	for epoch in 1:epochs
		for batch_cpu in train_loader
			x, y = batch_cpu |> device	# transfer data to device
			loss, grads = Flux.withgradient(model) do m
				predicted_logits = m(x)
				loss_function(predicted_logits, y, agg=x->Flux.mean(weights .* x))
			end
			push!(loss_history, loss)
			Flux.update!(optimizer_state, model, grads[1]) # update parameters pf model
		end
		println("Epoch $epoch | latest loss: $(round(loss_history[end], digits=4))")
	end
	
	return loss_history
end

"""
    confusion_matrix(y_n, y)

Compute a confusion matrix over `label_names` from predicted and true class-name vectors.

# Arguments
- `y_n`: vector of predicted class names (elements of `label_names`).
- `y`: vector of true class names (elements of `label_names`), same length as `y_n`.

# Returns
A `(length(label_names), length(label_names))` `Float64` matrix `cm` where
`cm[i, j]` is the count of samples predicted as `label_names[i]` whose true class is
`label_names[j]` (i.e. rows = predictions, columns = ground truth).

# Example
```julia
conf_mat = confusion_matrix(["normal", "pneumonia"], ["normal", "normal"])
```
"""
function confusion_matrix(ŷ, y)
	n_labels = length(label_names)
	conf_mat = zeros(Float64, (n_labels, n_labels))
	for (i, lbl_i)=enumerate(label_names), (j, lbl_j)=enumerate(label_names)
		 conf_mat[i, j] = sum((ŷ .== lbl_i) .&& (y .== lbl_j))
			end
	return conf_mat
end

"""
    per_class_accuracy(conf_mat)

Compute and print per-class accuracy from a confusion matrix produced by
[`confusion_matrix`](@ref).

# Arguments
- `conf_mat`: a square matrix where `conf_mat[i, j]` is the count predicted as class `i`
  with true class `j` (rows = predictions, columns = ground truth), indexed like `label_names`.

# Returns
A `Vector` of per-class accuracies (correct / total-true), in `label_names` order.
Also prints a formatted summary line per class as a side effect.
"""
function per_class_accuracy(conf_mat)
	class_totals = vec(sum(conf_mat, dims=1))       # total true instances per class (column sums)
	class_correct = diag(conf_mat)                  # correctly predicted per class (diagonal)
	accuracies = class_correct ./ class_totals
	for (name, acc) in zip(label_names, accuracies)
		println("  $(rpad(name, 22)) $(round(acc * 100, digits=1))%")
	end
	return accuracies
end

#plotting functions
"""
    plot_training_loss_history(loss_history, filename="pneumonia_training_loss_history.png", outdir="plots")

Plot per-batch training loss over time and save it as a PNG.

# Arguments
- `loss_history`: a vector of loss values, e.g. as returned by [`train_cnn!`](@ref).
- `filename::AbstractString="pneumonia_training_loss_history.png"`: output file name.
- `outdir::AbstractString="plots"`: output directory; created via `mkpath` if it doesn't exist.

# Returns
`nothing`. As a side effect, saves the figure to `outdir/filename` and prints a confirmation message.
"""
function plot_training_loss_history(loss_history, filename="pneumonia_training_loss_history.png", outdir="plots")

	training_history_figure = Figure(size=(450,450))
	ax = CairoMakie.Axis(
		training_history_figure[1,1]; 
		xlabel="training steps", 
		ylabel="loss",
		title="Training history",
	)
	
	lines!(ax, loss_history)

	mkpath(outdir)
	save(joinpath(outdir, filename), training_history_figure)
	println("Saved training loss history to $outdir/$filename")
end

"""
    plot_confusion_matrix(conf_mat, filename="pneumonia_confusion_matrix.png", outdir="plots")

Render a confusion matrix as an annotated heatmap and save it as a PNG.

# Arguments
- `conf_mat`: a square matrix as produced by [`confusion_matrix`](@ref), indexed like `label_names`.
- `filename::AbstractString="pneumonia_confusion_matrix.png"`: output file name.
- `outdir::AbstractString="plots"`: output directory; created via `mkpath` if it doesn't exist.

# Returns
`nothing`. As a side effect, saves the figure to `outdir/filename` and prints a confirmation message.
"""
function plot_confusion_matrix(conf_mat, filename="pneumonia_confusion_matrix.png", outdir="plots")
		# Create a figure
		conf_mat_figure = Figure(size = (600, 600))
		
		# Create an axis for the heatmap
		conf_mat_axis = Axis(
			conf_mat_figure[1, 1], 
			aspect=1,
		    title = "Confusion Matrix", 
		    xlabel = "predictions", 
		    ylabel = "ground truth",
			xticks = (1:length(label_names), label_names),
			xticklabelrotation = pi/2,
			yticks = (1:length(label_names), label_names)
		)
		
		# Plot the heatmap
		hm = heatmap!(
			conf_mat_axis, 
			1:size(conf_mat, 2), 
			1:size(conf_mat, 1), 
			conf_mat, 
			colormap = :viridis
		)
		
		# # Annotate the heatmap
		for i in 1:length(label_names)
		    for j in 1:length(label_names)
		        text!(conf_mat_axis, i, j, text = string(Int(conf_mat[i, j])), align = (:center, :center), color = :gray, fontsize = 20)
		    end
		end

		Colorbar(conf_mat_figure[1, 2], hm)

		mkpath(outdir)
		save(joinpath(outdir, filename), conf_mat_figure)
		println("Saved confusion matrix to $outdir/$filename")

end

"""
    extract_latent_space(model, x_batch; device=cpu)

Run `x_batch` through every layer of `model` except the final classification layer,
returning the resulting feature activations. Used to obtain a latent representation
for visualization (see [`pca_2d`](@ref) and [`plot_latent_space`](@ref)).

Temporarily switches `model` to test mode (so `BatchNorm` uses running statistics
rather than batch statistics) and restores train mode before returning.

# Arguments
- `model`: a `Flux.Chain` (e.g. from [`build_cnn`](@ref)) whose last layer is the
  classification head.
- `x_batch`: a batch of inputs in `(H, W, C, N)` layout.

# Keyword Arguments
- `device=cpu`: device to run the forward pass on; the result is always moved back to `cpu`.

# Returns
A `(features, N)` array of activations from the penultimate layer, one column per sample.
"""
function extract_latent_space(model, x_batch; device=cpu)
	Flux.testmode!(model)  # same BatchNorm reasoning as saliency maps

	# everything except the final classification layer
	feature_extractor = Chain(model.layers[1:end-1]...)
	activations = feature_extractor(x_batch |> device) |> cpu

	Flux.trainmode!(model)
	return activations  
end

"""
    pca_2d(activations)

Project feature activations onto their top 2 principal components via SVD, for
2D visualization of a latent space.

# Arguments
- `activations`: a `(features, N)` array, e.g. as returned by [`extract_latent_space`](@ref).

# Returns
A `(2, N)` array of coordinates: row 1 is the projection onto the first principal
component (PC1, highest variance), row 2 onto the second (PC2).
"""
function pca_2d(activations)
	# activations: (features, N)
	mean_vec = vec(mean(activations, dims=2))
	centered = activations .- mean_vec

	U, S, Vt = svd(centered)
	pcs = U[:, 1:2]' * centered  # project onto top 2 principal components → (2, N)
	return pcs
end

"""
    plot_latent_space(coords, labels; filename=nothing)

Scatter-plot 2D latent coordinates colored/grouped by class label, e.g. the output
of [`pca_2d`](@ref).

# Arguments
- `coords`: a `(2, N)` array of 2D coordinates (row 1 = x/PC1, row 2 = y/PC2).
- `labels`: a length-`N` vector of class labels (one per sample); each unique value
  becomes a separate legend entry.

# Keyword Arguments
- `filename=nothing`: if given, the figure is also saved to this path via `save`.

# Returns
The `CairoMakie.Figure` object, so it can be displayed (e.g. in Pluto) or further customized.
"""
function plot_latent_space(coords, labels; filename="latent_space_pca.png", outdir="plots")
	fig = Figure(size=(600, 600))
	ax = CairoMakie.Axis(fig[1,1], title="Latent space (PCA)", xlabel = "PC1", ylabel = "PC2",aspect=1)

	for lbl in unique(labels)
		idx = findall(==(lbl), labels)
		scatter!(ax, coords[1, idx], coords[2, idx], label=lbl, markersize=8)
	end
	axislegend(ax)
	mkpath(outdir)
	save(joinpath(outdir, filename), fig)
	println("Saved plot to $outdir/$filename")
	return fig
end

function plot_latent_space_errors(coords, y_hat, y; filename="latent_space_errors.png", outdir="plots")
	# coords: (2, N) from pca_2d
	# y_hat, y: predicted and true label names (same format as confusion_matrix)
	correct = y_hat .== y

	fig = Figure(size=(600, 600))
	ax = CairoMakie.Axis(fig[1,1],
		title = "Latent space: correct vs. misclassified",
		xlabel = "PC1", ylabel = "PC2", aspect = 1)

	# correct predictions: faint, colored by true class
	for lbl in unique(y)
		idx = findall(correct .&& (y .== lbl))
		scatter!(ax, coords[1, idx], coords[2, idx],
			label = "$lbl (correct)", markersize = 8, alpha = 0.35)
	end

	# misclassified: drawn last so they sit on top
	for lbl in unique(y)
		idx = findall(.!correct .&& (y .== lbl))
		isempty(idx) && continue
		scatter!(ax, coords[1, idx], coords[2, idx],
			label = "$lbl (misclassified)", markersize = 12,
			marker = :xcross, strokewidth = 1.5)
	end

	axislegend(ax)
	mkpath(outdir)
	save(joinpath(outdir, filename), fig)
	println("Saved plot to $outdir/$filename")
	return fig
end

"""
    saliency_map(m, x; class_idx=nothing)
 
Compute a vanilla-gradient saliency map for a single input `x`, showing which pixels
most influence the model's prediction for a given class. Temporarily switches `m` to
test mode (so `BatchNorm` uses running statistics) but does **not** restore train mode
before returning — call `Flux.trainmode!(m)` afterward if you need `m` back in
training mode.
 
# Arguments
- `m`: a `Flux.Chain` (e.g. from [`build_cnn`](@ref)).
- `x`: a single-sample batch in `(H, W, C, 1)` layout.
 
# Keyword Arguments
- `class_idx=nothing`: index of the class to compute saliency for. If `nothing`, uses
  the model's predicted class (`Flux.onecold(logits)[1]`).
 
# Returns
A `(H, W)` `Float32` array of saliency values, normalized to `[0, 1]` (channels are
collapsed by taking the maximum absolute gradient across channels at each pixel).
"""
function saliency_map(m, x; class_idx=nothing)
		Flux.testmode!(m)

		logits = m(x)
		target_idx = class_idx === nothing ? Flux.onecold(logits)[1] : class_idx 

		num_classes = size(logits, 1)
		selector = Float32.((1:num_classes) .== target_idx) |> cpu

		grads = Zygote.gradient(x) do input
			out = m(input)
			sum(out[:, 1] .* selector)
		end

		sal = abs.(grads[1][:, :, :, 1])              # (H, W, C)
		sal = dropdims(maximum(sal, dims=3), dims=3)   # collapse channels -> (H,W), basically taking the value with the highest RGB value
		sal = sal ./ (maximum(sal) + 1f-8)
		return sal
	end

"""
    smoothgrad(m, x; class_idx=nothing, n=50, noise_level=0.15f0)
 
Compute a SmoothGrad saliency map: averages [`saliency_map`](@ref) over `n` noisy
copies of `x`, which typically produces a less visually noisy attribution map than
a single vanilla-gradient saliency map.
 
# Arguments
- `m`: a `Flux.Chain` (e.g. from [`build_cnn`](@ref)).
- `x`: a single-sample batch in `(H, W, C, 1)` layout.
 
# Keyword Arguments
- `class_idx=nothing`: index of the class to compute saliency for. If `nothing`, uses
  the model's predicted class on the original (noise-free) `x`; that same class index
  is then held fixed across all `n` noisy samples.
- `n::Integer=50`: number of noisy samples to average over.
- `noise_level::Float32=0.15f0`: standard deviation of the Gaussian noise added to `x`,
  expressed as a fraction of `x`'s value range (`maximum(x) - minimum(x)`).
 
# Returns
A `(H, W)` `Float32` array of averaged saliency values, normalized to `[0, 1]`.
"""
function smoothgrad(m, x; class_idx=nothing, n=50, noise_level=0.15f0)
		Flux.testmode!(m)

		logits = m(x)
		target = class_idx === nothing ? Flux.onecold(logits)[1] : class_idx
		sigma = noise_level * (maximum(x) - minimum(x))

		acc = zeros(Float32, size(x, 1), size(x, 2))
		for _ in 1:n
			xn = x .+ sigma .*randn(Float32, size(x))
			acc .+= saliency_map(m, xn; class_idx = target)
		end
		acc ./= n
		return acc ./ (maximum(acc) + 1f-8)
	end

"""
    arr_to_img(arr)
 
Convert a raw `(H, W, C)` array (e.g. a single image from a preprocessed batch, or a
saliency map broadcast to match channel count) into a Colors.jl image object suitable
for display or saving.
 
# Arguments
- `arr`: a 3D array of size `(H, W, 1)` (grayscale) or `(H, W, 3)` (RGB).
 
# Returns
A `Gray`-valued image (`(H, W)` array of `Gray` for 1-channel input) or an `RGB`-valued
image (`(H, W)` array of `RGB` for 3-channel input).
"""
function arr_to_img(arr)
	@assert ndims(arr) == 3 "Expected a height × width × channels array"
	@assert size(arr, 3) in (1, 3) "Expected 1 (grayscale) or 3 (RGB) channels"
 
	if size(arr, 3) == 1
		return Gray.(arr[:, :, 1])
	else
		return RGB.(arr[:, :, 1], arr[:, :, 2], arr[:, :, 3])
	end
end

"""
    alpha_cmap(cm=:inferno; amax=0.75f0, γ=1.5f0)
 
Build a colormap with a variable alpha (transparency) ramp, useful for overlaying a
saliency map ([`saliency_map`](@ref) / [`smoothgrad`](@ref)) on top of the original
image: low-saliency regions become nearly transparent while high-saliency regions
stay opaque, so the underlying image remains visible through the overlay.
 
# Arguments
- `cm=:inferno`: base colormap name/symbol, resolved via `Makie.to_colormap`.
 
# Keyword Arguments
- `amax::Float32=0.75f0`: maximum alpha (opacity) at the top of the colormap; `1.0`
  would make the most intense color fully opaque.
- `γ::Float32=1.5f0`: gamma exponent applied to the alpha ramp. Values above `1`
  concentrate transparency at the low end (more of the colormap stays faint before
  ramping up); values below `1` do the opposite.
 
# Returns
A `Vector{RGBAf}` of colors with the same length and RGB values as `to_colormap(cm)`,
but with alpha values ramped from `0` to `amax` following the `γ` power curve.
"""
function alpha_cmap(cm = :inferno; amax = 0.75f0, γ = 1.5f0)
    cols = to_colormap(cm)
    n = length(cols)
    [RGBAf(c.r, c.g, c.b, amax * ((i - 1) / (n - 1))^γ) for (i, c) in enumerate(cols)]
end



end # module CNNpneumonia