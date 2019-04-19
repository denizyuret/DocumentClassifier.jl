# DocumentClassifier.jl

The following scripts should be run in the given order:

* setup.jl: installs required packages and sets constants
* dict.jl: creates a dictionary and saves it to dict.jld2
* data.jl: shuffles, splits, minibatches data and saves it to data.jld2
* rnn.jl: trains simple RNN classifier with ~0.86 validation accuracy and saves to rnn.jld2
