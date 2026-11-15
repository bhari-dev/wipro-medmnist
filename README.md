# MedMNIST Explainability - Group 22

A Julia implementation of a CNN for medical image classification using
the MNIST dataset and the Flux machine learning library, paired with Explainable AI techniques to understand how and why the model classifies the images

This repository belongs to group 22, consisting of the following members:

* Aman Singh Bhari
* Leo Lindenberg
* Sopheavisal Lim

This repository serves for the group project in the course
Scientific Programming by Dr Tobias Knopp and Dr Martin Möddel at the Technical University Hamburg Harburg (TUHH).

## Background

Medical image classification plays a vital role in modern-day health care. It assists doctors in diagnosing X-rays and ultrasounds. Deep neural networks have achieved remarkable results in image classification but have been labelled as “black boxes”. Since knowing what influenced a neural network's decision is vital, implementing Explainable AI (XAI) techniques helps users understand why the network made that prediction. This project trains a CNN on the PneumoniaMNIST dataset and applies XAI techniques to understand why the network makes a given prediction, rather than treating it purely as a black box.

### Dataset

As mentioned above, we chose to work with the PneumoniaMNIST dataset. PneumoniaMNIST is a binary dataset (normal vs pneumonia classes) consisting of chest X-ray images as part of the MedMNIST v2 collection. Note that other datasets can also be used, e.g BreastMNIST, but for this project we chose to focus on PneumoniaMNIST. 

## Project Description

We were tasked with building a classifier for a MedMNIST subset and using Explainable AI techniques to make its decision-making process more transparent, rather than treating the classifier as a black box.

For this project, we mainly worked with PneumoniaMNIST, however, we experimented with other datasets as mentioned above. 

* Our goal was to build an accurate CNN and understand why it makes each prediction the way it does.

* We started with a straightforward CNN: three convolutional blocks, flattening into two dense layers.
This alone achieved at least 85% test accuracy, but we noticed the model was consistently weaker on one class than the other because PneumoniaMNIST is imbalanced toward pneumonia cases. By increasing the decision threshold and implementing inverse-frequency class weights, we fixed this problem and reached an accuracy of 90.5% with balanced class accuracies.

* Once we reached our target accuracy of 90.5%, we froze the model’s weights, saving them to a BSON file so that we can work with the desired model later on

* With the classifier settled, we turned to explainability. We implemented saliency maps to visualise which regions of an X-ray the model actually relied on for its decision. The vanilla saliency maps were noisy, so we implemented Smooth-Grad and overlay for better understanding and clearer visualisation.

* Finally, we investigated the network’s latent space by extracting 128-dimensional activations from the penultimate layer for a batch of test images and projecting them to 2D with PCA to see how cleanly the network separates normal and pneumonia cases internally.

* With that, we had successfully achieved our goal of this project.

## Project Structure

We structured our project the following way:

```
.
├── model/                       # frozen model weights (generated after training)
├── plots/                       # generated figures (confusion matrix, saliency maps, etc.)
├── presentation/                # contains notebook presentation for the course
├── scripts/
│   └── download_data.py         # downloads the PneumoniaMNIST dataset using a Python script
├── src/
│   ├── breast/                  # experimented data sets
│   ├── derma/                   # experimented data sets
│   ├── CNNpneumonia.jl          # core module: model, training, evaluation, XAI methods
│   ├── train_cnn_pneumonia.jl   # training + evaluation entry point
│   ├── saliency.jl              # notebook for saliency map generation for individual predictions 
│   ├── latent_space.jl          # latent space extraction + PCA/t-SNE visualization
├── tests/
│   └── runtests.jl              # unit tests for core methods
│   └── test_laten_space.jl      # checks output shape and that PC1 has more variance than PC2 on synthetic data engineered to have that property.
│   └── test_metrics.jl          # checks confusion_matrix and per_class_accuracy against a hand-computed example each
│   └── test_model.jl            # mostly checks shapes (model isa Flux.Chain, output size, loss history is finite/non-empty), not that the model is learning correctly.
│   └── test_plotting.jl         # checks that a Figure object comes back and a file gets written, not that the plot looks right.
│   └── Project.toml
│   └── Manifest.toml
├── Project.toml
├── Manifest.toml
├── Roadmap.md                   # project timeline and milestones
└── README.md

```
## Installation

```julia
# from the project root
julia --project=.
julia> using Pkg; Pkg.instantiate()
```

## Downloading the data

To download the dataset, you can use our Python script `download_data.py`, which can be found in the scripts folder. 

Usage:
      `python scripts/download_data.py [dataset] [--size {28,64,128,224}]`

If the size is not specified, 64 is the default size.
If no dataset is given on the command line, you'll be prompted to pick one
from a menu. The data files are git-ignored; run this script once per
dataset after cloning. 

## Usage

All commands should be run from the repository root, so that --project=.
resolves to the correct Project.toml

### Building & Training the model

```julia
julia --project=. src/train_cnn_pneumonia.jl
```
Upon running this command, the following will happen:

* load and preprocess the data
* train a CNN with class-weighted loss to handle class imbalance
* evaluate on the held-out test set (overall + per-class accuracy, confusion matrix)
* save training loss and confusion matrix plots to plots/
* freeze and save the model to model_frozen_accuracy.bson 

### Generating Saliency maps

```
julia --project=. src/saliency.jl
```

Loads the frozen model and computes the gradient of the correct class's score with
respect to input pixels for a chosen test image, visualising which pixels most
influenced the prediction. Output saved to plots/

### Visualise the latent space

```
julia --project=. src/latent_space.jl
```
Extracts activations from the second-to-last (128-dim) layer for a batch of test
images and projects them to 2D via PCA, to visualise how the network clusters
different medical conditions. Output saved to plots/

### Running the tests (in Julia)

```julia
] activate .
test
```

or, from within the `test/` environment:

```julia
julia --project=test test/runtests.jl
```

`test/Project.toml` declares its own dependencies (`CairoMakie`, `Flux`, `LinearAlgebra`,
`Statistics`, `Test`), separate from the main `Project.toml` — run
`julia --project=test -e 'using Pkg; Pkg.instantiate()'` once before testing if you haven't
already. The suite covers model construction/training, metrics, latent-space
extraction/PCA, and plotting. See `test/*.jl` for details.

> Note: tests that reference `Figure` (in `test_plotting.jl`) require `using CairoMakie`
> at the top of `runtests.jl`. It's a dependency of `CNNpneumonia`, but that `using` lives
> inside the module's own scope, so its exports aren't automatically visible in `Main`
> where the test files run.

## Doc strings

All public functions in `CNNpneumonia.jl` are documented with Julia docstrings. Once the
package is loaded, to view them from the REPL do:

```julia
] activate .
using CNNpneumonia
?build_cnn
?train_cnn!
```
Documented functions:

| Function | Purpose |
|---|---|
| `get_data` | Load and preprocess a MedMNIST-style `.npz` dataset |
| `build_cnn` | Construct the CNN architecture |
| `train_cnn!` | Train a model in place |
| `label_names` | Ordered class names (`["normal", "pneumonia"]`) |
| `confusion_matrix` | Compute a confusion matrix |
| `per_class_accuracy` | Compute/print per-class accuracy |
| `plot_training_loss_history` | Plot & save training loss curve |
| `plot_confusion_matrix` | Plot & save confusion matrix heatmap |
| `extract_latent_space` | Extract penultimate-layer activations |
| `pca_2d` | Project activations to 2D via PCA |
| `plot_latent_space` | Scatter-plot a 2D latent space by class |

## References

* Flux.jl Documentation

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778

* Medmnist