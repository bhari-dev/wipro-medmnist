module CNNbreast

using Flux, Flux.Losses, CUDA, 
		MLDatasets, PlutoUI, CairoMakie, 
		LinearAlgebra, FileIO, OneHotArrays,
		Downloads, NPZ, Colors, ImageBase
	
	CairoMakie.activate!(; px_per_unit = 4)

using Statistics: mean
using LinearAlgebra: svd

export get_data, build_cnn, train_cnn!, label_names,
	confusion_matrix, per_class_accuracy, find_best_threshold,
	plot_training_loss_history, plot_confusion_matrix,
	extract_latent_space, pca_2d, plot_latent_space, plot_latent_space_errors




const DATA_FOLDER = joinpath(@__DIR__,"..","data")

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

label_names = ["malignant", "normal, benign"]


function augment_batch(x)
	x = rand() > 0.5 ? x[:, end:-1:1, :, :] : x       # horizontal flip
	x = rand() > 0.5 ? x[end:-1:1, :, :, :] : x       # vertical flip
	brightness = Float32(rand() * 0.2 - 0.1)           # uniform in [-0.1, 0.1]
	contrast   = Float32(rand() * 0.4 + 0.8)           # uniform in [0.8, 1.2]
	return clamp.(x .* contrast .+ brightness, 0f0, 1f0)
end


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

		Dropout(0.5),
		Dense(4096 => 128, relu),
		Dropout(0.5),
		Dense(128 => num_classes),
	)

	return cnn
end


function train_cnn!(model, train_data, val_data;
	epochs=50, batchsize=32,
	device=cpu,
	optimizer=AdamW(1e-3, (0.9, 0.999), 1e-4),
	weights=ones(Float32, length(label_names)) |> device,
	loss_function=logitcrossentropy)

	train_loader = Flux.DataLoader(train_data, batchsize=batchsize, shuffle=true)
	optimizer_state = Flux.setup(optimizer, model)
	val_x, val_y = val_data

	loss_history = []
	for epoch in 1:epochs
		Flux.trainmode!(model)
		for (x_cpu, y_cpu) in train_loader
			x = augment_batch(x_cpu) |> device
			y = y_cpu |> device
			loss, grads = Flux.withgradient(model) do m
				loss_function(m(x), y, agg=v->Flux.mean(weights .* v))
			end
			push!(loss_history, loss)
			Flux.update!(optimizer_state, model, grads[1])
		end

		# validation
		Flux.testmode!(model)
		val_logits = model(val_x |> device) |> cpu
		val_loss   = loss_function(val_logits, val_y, agg=v->Flux.mean(weights .* v))
		val_acc    = mean(Flux.onecold(val_logits) .== Flux.onecold(val_y))
		println("Epoch $epoch | train_loss: $(round(loss_history[end], digits=4)) | val_loss: $(round(Float32(val_loss), digits=4)) | val_acc: $(round(val_acc*100, digits=1))%")
	end

	return loss_history
end

function confusion_matrix(ŷ, y)
	n_labels = length(label_names)
	conf_mat = zeros(Float64, (n_labels, n_labels))
	for (i, lbl_i)=enumerate(label_names), (j, lbl_j)=enumerate(label_names)
		 conf_mat[i, j] = sum((ŷ .== lbl_i) .&& (y .== lbl_j))
			end
	return conf_mat
end

function per_class_accuracy(conf_mat)
	class_totals = vec(sum(conf_mat, dims=1))       # total true instances per class (column sums)
	class_correct = diag(conf_mat)                  # correctly predicted per class (diagonal)
	accuracies = class_correct ./ class_totals
	for (name, acc) in zip(label_names, accuracies)
		println("  $(rpad(name, 22)) $(round(acc * 100, digits=1))%")
	end
	return accuracies
end

function find_best_threshold(model, x_val, y_val; device=cpu)
	Flux.testmode!(model)
	probs = softmax(model(x_val |> device) |> cpu)   # (2, N): row 1 = P(malignant)
	Flux.trainmode!(model)
	true_labels = onecold(y_val, label_names)

	best_threshold, best_bal_acc = 0.5f0, 0.0
	for t in range(0.2, 0.8, step=0.05)
		preds = [probs[1, i] > t ? label_names[1] : label_names[2] for i in axes(probs, 2)]
		conf  = confusion_matrix(preds, true_labels)
		totals = vec(sum(conf, dims=1))
		bal_acc = mean(diag(conf) ./ max.(totals, 1.0))
		if bal_acc > best_bal_acc
			best_bal_acc = bal_acc
			best_threshold = Float32(t)
		end
	end
	println("Best threshold: $(round(best_threshold, digits=2)) → val balanced accuracy: $(round(best_bal_acc*100, digits=1))%")
	return best_threshold
end


function plot_training_loss_history(loss_history, filename="breast_training_loss_history.png", outdir="plots")

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


function plot_confusion_matrix(conf_mat, filename="breast_confusion_matrix.png", outdir="plots")
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


function extract_latent_space(model, x_batch; device=cpu)
	Flux.testmode!(model)  # same BatchNorm reasoning as saliency maps

	# everything except the final classification layer
	feature_extractor = Chain(model.layers[1:end-1]...)
	activations = feature_extractor(x_batch |> device) |> cpu

	Flux.trainmode!(model)
	return activations  
end

function pca_2d(activations)
	# activations: (features, N)
	mean_vec = vec(mean(activations, dims=2))
	centered = activations .- mean_vec

	U, S, Vt = svd(centered)
	pcs = U[:, 1:2]' * centered  # project onto top 2 principal components → (2, N)
	return pcs
end

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
end

function plot_latent_space_errors(coords, y_hat, y; filename="latent_space_errors.png", outdir="plots")
	# coords: (2, N) from pca_2d
	# ŷ, y: predicted and true label names (same format as confusion_matrix)
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
end

end # module CNNbreast

