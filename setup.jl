# Install packages, set constants

using Pkg
for p in ("Knet","CSV","FileIO","JLD2","WordTokenizers","IterTools")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

using WordTokenizers, Knet
set_tokenizer(poormans_tokenize) # removes all punctuation, splits on space, should we lowercase?
Knet.seed!(1)                    # to get replicable results
ENV["COLUMNS"]=80                # display width

#DATAFILE = "ML_Dataset_20190227.csv"
DATAFILE = "news_dataset_20190415"
MAXLEN = 256  # maximum size of the word sequence, pad shorter sequences, truncate longer ones
VOCAB = 65535 # maximum vocabulary size, keep the most frequent VOCAB words, map the rest to UNK token
PAD = VOCAB   # pad token
UNK = VOCAB-1 # unk token
TRAIN = 0.8   # ratio of training examples
VALID = 0.1   # ratio of validation examples
TEST = 0.1    # ratio of test examples
BATCH = 100   # Number of instances in a minibatch
RNNTYPE=:gru  # :lstm, :gru, :relu, :tanh
EMBED=256     # Word embedding size
HIDDEN=256    # Hidden layer size
EPOCHS=2      # Number of training epochs
DROPOUT=0.5   # Dropout rate

nothing
