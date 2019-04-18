using Knet, FileIO, Statistics, IterTools

# Set constants for the model and training
VOCABSIZE=65535   # maximum vocabulary size, keep the most frequent 64K, map the rest to UNK token
EMBEDSIZE=256     # Word embedding size
RNNTYPE=:gru      # :lstm, :gru, :relu, :tanh
NUMHIDDEN=256     # Hidden layer size
NUMCLASS=9        # number of output classes

EPOCHS=2          # Number of training epochs
DROPOUT=0.5       # Dropout rate

UNK = VOCABSIZE   # Unknown token
PAD = VOCABSIZE-1 # Pad token
ENV["COLUMNS"]=80 # Display width

# Load and minibatch data
(wcnt,words,w2i,classes) = load("dict.jld2","count","words","w2i","classes")
(dtrn,dval,dtst) = load("data.jld2","dtrn","dval","dtst")
@show length.((dtrn,dval,dtst))

# Define functions that can print the actual words and do individual predictions:
pwords = copy(words); pwords[PAD]="<pad>"; pwords[UNK]="<unk>";
printwords(x,y=0)=(x = x[x.!=PAD]; """$(classes[y])\n$(join(pwords[x]," "))""")
predict(model,x)="\nPrediction: " * classes[argmax(Array(vec(model([x]))))]

# Define model
struct Embed; w; end
Embed(vocab::Int, embed::Int) = Embed(param(embed,vocab))
(e::Embed)(x) = e.w[:,permutedims(hcat(x...))]  # [T,T,...,T]->hcat->(T,B)->pdim->(B,T)->emb->(X,B,T)->rnn->(H,B,T)

struct Output; w; b; end
Output(input::Int, output::Int)=Output(param(output,input), param0(output))
(l::Output)(x) = l.w * x[:,:,end] .+ l.b  # (H,B,T)->(H,B)->(C,B)

struct Dropout; p; end
(d::Dropout)(x) = dropout(x,d.p)

struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Knet.Data) = mean(c(x,y) for (x,y) in d)

SequenceClassifier(input::Int, embed::Int, hidden::Int, output::Int; pdrop=0, rnnType=:lstm) =
    Chain((Embed(input,embed),
           Dropout(pdrop),
           RNN(embed,hidden,rnnType=RNNTYPE),
           Dropout(pdrop),
           Output(hidden,output)))

# Clean up last run
model = nothing; Knet.gc()
# Initialize model
model = SequenceClassifier(VOCABSIZE,EMBEDSIZE,NUMHIDDEN,NUMCLASS,pdrop=DROPOUT,rnnType=RNNTYPE)
# Use adam for optimization -- note that this returns an iterator and does not run it yet
opt = adam(model,repeat(dtrn,EPOCHS))
# Run the optimizer and report validation accuracy every 80 iterations
progress!(accuracy(model,dval) for _ in takenth(opt,80))

# Display final accuracy
@show accuracy(model,dtst)
@show accuracy(model,dval)
@show accuracy(model,dtrn)
