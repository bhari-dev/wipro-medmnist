module CNNderma

using Flux, Flux.Losses, CUDA, 
		MLDatasets, PlutoUI, CairoMakie, 
		LinearAlgebra, FileIO, OneHotArrays,
		Downloads, NPZ, Colors, ImageBase
	
	CairoMakie.activate!(; px_per_unit = 4)

export get_data, build_cnn, train_cnn!, label_names,
	confusion_matrix, per_class_accuracy, plot_training_loss_history, plot_confusion_matrix

const DATA_FOLDER = joinpath(@__DIR__, "../","data")

function get_data(dataset_name, resolution=64)
	dataset_name = lowercase(dataset_name)
	data_path = joinpath(DATA_FOLDER, dataset_name, "$(dataset_name)_$(resolution).npz")
	if !isfile(data_path)
		@info "The file $data_path does not exists, please use the script to download the data"
	end
	data = npzread(data_path)
	labels = minimum(data["train_labels"]):maximum(data["train_labels"])
	img_preprocess_fn = img -> permutedims(img, (2:ndims(img)..., 1))
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

label_names = ["a. k. i. c.", #actinic keratoses and intraepithelial carcinoma
			   "b. c. c.",    #basal cell carcinoma
			   "b. k. l.",    #benign keratosis-like lesions
			   "dermatofibroma",
			   "melanoma",
			   "melanocytic nevi",
			   "vascular lesions"]



function build_cnn(; imagesize, num_classes)
	cnn = Chain(
		Conv((3, 3), 3 => 16, relu, pad=1),
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


function train_cnn!(model, train_data, test_data; 
	epochs=2, batchsize=128, 
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

#plotting functions

function plot_training_loss_history(loss_history, filename="training_loss_history.png", outdir="plots")

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

function plot_confusion_matrix(conf_mat, filename="confusion_matrix.png", outdir="plots")
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

end # module CNNderma
