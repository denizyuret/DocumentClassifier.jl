using FileIO, CSV, WordTokenizers, Random, Knet
set_tokenizer(poormans_tokenize) # removes all punctuation, splits on space

CLASSES = Dict{String,UInt8}("entertainment" => 1, "health" => 2, "sports" => 3, "us" => 4, "world" => 5, "technology" => 6, "politics" => 7, "science" => 8, "business" => 9);
VOCAB = typemax(UInt16)
MAXLEN=256        # maximum size of the word sequence, pad shorter sequences, truncate longer ones
BATCHSIZE=100     # Number of instances in a minibatch
UNK = VOCAB
PAD = VOCAB-1
NDATA = 180000
TRAIN = 160000
VALID = 10000
TEST = 10000
W2I = load("dict.jld2", "w2i")

function row2xy(r)
    y = CLASSES[r.category]
    x = Array{UInt16}(undef,MAXLEN)
    x .= PAD
    i = 0
    for w in tokenize(r.text)
        i += 1
        i > MAXLEN && break
        w2i = W2I[w]
        w2i > VOCAB && (w2i = UNK)
        x[i] = w2i
    end
    (x,y)
end

xdata = Array{UInt16}[]
ydata = UInt8[]

for r in CSV.File("ML_Dataset_20190227.csv")
    (x,y) = row2xy(r)
    push!(xdata,x)
    push!(ydata,y)
end

# shuffle and split and minibatch

Random.seed!(1)
r = randperm(length(xdata))
xtrn = xdata[r[1:TRAIN]]
ytrn = ydata[r[1:TRAIN]]
xval = xdata[r[1+TRAIN:VALID+TRAIN]]
yval = ydata[r[1+TRAIN:VALID+TRAIN]]
xtst = xdata[r[1+VALID+TRAIN:TEST+VALID+TRAIN]]
ytst = ydata[r[1+VALID+TRAIN:TEST+VALID+TRAIN]]
dtrn = minibatch(xtrn,ytrn,BATCHSIZE)
dval = minibatch(xval,yval,BATCHSIZE)
dtst = minibatch(xtst,ytst,BATCHSIZE)
save("data.jld2", "dtrn", dtrn, "dval", dval, "dtst", dtst)
