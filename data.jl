include("setup.jl")
using FileIO, CSV, WordTokenizers, Random, Knet

@info "Reading dict.jld2"
(widx,cidx) = load("dict.jld2", "widx", "cidx")

@info "Reading $DATAFILE"
xdata = Array{UInt16}[]
ydata = UInt8[]
for r in CSV.File(DATAFILE)
    y = cidx[r.category]
    x = Array{UInt16}(undef,MAXLEN)
    x .= PAD
    i = 0
    for t in tokenize(r.text)
        i += 1
        i > MAXLEN && break
        w = widx[t]
        w > VOCAB && (w = UNK)
        x[i] = w
    end
    push!(xdata,x)
    push!(ydata,y)
end

@info "Shuffle, split, minibatch $(length(xdata)) instances"
n = length(xdata)
r = randperm(n)
ntrn = round(Int,n*TRAIN)
nval = round(Int,n*VALID)
ntst = round(Int,n*TEST)
xtrn = xdata[r[1:ntrn]]
ytrn = ydata[r[1:ntrn]]
xval = xdata[r[1+ntrn:nval+ntrn]]
yval = ydata[r[1+ntrn:nval+ntrn]]
xtst = xdata[r[1+nval+ntrn:end]]
ytst = ydata[r[1+nval+ntrn:end]]
dtrn = minibatch(xtrn,ytrn,BATCH)
dval = minibatch(xval,yval,BATCH)
dtst = minibatch(xtst,ytst,BATCH)

@info "Saving data.jld2"
save("data.jld2", "dtrn", dtrn, "dval", dval, "dtst", dtst)
