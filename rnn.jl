include("setup.jl")
using Knet, FileIO, Statistics, IterTools

# Load and minibatch data
@info "Reading dict.jld2"
(wcnt,words,widx,classes) = load("dict.jld2","wcnt","words","widx","classes")
@info "Reading data.jld2"
(dtrn,dval,dtst) = load("data.jld2","dtrn","dval","dtst")
@show CLASS = length(classes)
@show length.((dtrn,dval,dtst))

# Define functions that can print the actual words and do individual predictions for debugging:
pwords = words[1:VOCAB]; pwords[PAD]="<pad>"; pwords[UNK]="<unk>";
printwords(x)=(x = x[x.!=PAD]; println(join(pwords[x]," ")))
predict(model,x)=println(classes[argmax(Array(vec(model([x]))))])

# Define model
struct Embed; w; end
Embed(vocab::Int, embed::Int) = Embed(param(embed,vocab))
(e::Embed)(x) = e.w[:,permutedims(hcat(x...))]  

struct Output; w; b; end
Output(input::Int, output::Int)=Output(param(output,input), param0(output))
(l::Output)(x) = l.w * x[:,:,end] .+ l.b

struct Dropout; p; end
(d::Dropout)(x) = dropout(x,d.p)

struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Knet.Data) = mean(c(x,y) for (x,y) in d)

SequenceClassifier(input::Int, embed::Int, hidden::Int, output::Int; pdrop=DROPOUT, rnnType=RNNTYPE) =
    Chain((Embed(input,embed),
           Dropout(pdrop),
           RNN(embed,hidden,rnnType=rnnType),
           Dropout(pdrop),
           Output(hidden,output)))

# Note on Dimensions: T=MAXLEN, B=BATCH, X=EMBED, H=HIDDEN, C=CLASS
# [T,T,...,T]->hcat->(T,B)->pdim->(B,T)->emb->(X,B,T)->rnn->(H,B,T)->end->(H,B)->out->(C,B)

@info "Training for $EPOCHS epochs, monitoring validation accuracy"
# Clean up last run
bestscore = 0
bestmodel = model = nothing
Knet.gc()
# Initialize model
model = SequenceClassifier(VOCAB,EMBED,HIDDEN,CLASS)
# Monitor validation accuracy and keep best model
function monitor()
    global model, bestmodel, bestscore
    score=accuracy(model,dval)
    if score > bestscore
        bestmodel,bestscore = deepcopy(model),score
    end
    return score
end
# Use adam for optimization -- note that this returns an iterator and does not run it yet
opt = adam(model,repeat(dtrn,EPOCHS))
# Run the optimizer and monitor validation accuracy every 100 iterations
progress!(monitor() for _ in takenth(opt,100))

# Display final accuracy
@info "Computing final accuracy"
@show accuracy(bestmodel,dtst)
@show accuracy(bestmodel,dval)
@show accuracy(bestmodel,dtrn)

@info "Saving model to rnn.jld2"
Knet.save("rnn.jld2", "model", bestmodel)
