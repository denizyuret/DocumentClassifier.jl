include("setup.jl")
using FileIO, CSV, WordTokenizers

@info "Reading $DATAFILE"
f = CSV.File(DATAFILE)
d = Dict{String,UInt32}()
c = Dict{String,Int}()
for r in f
    c[r.category] = 1 + get(c,r.category,0)
    for w in tokenize(r.text)
        d[w] = 1 + get(d,w,0)
    end
end

@info "Found $(length(d)) unique words, $(length(c)) unique classes:\n$c"
words = sort(collect(keys(d)), by=(x->d[x]), rev=true)
classes = sort(collect(keys(c)))
widx = Dict{String,UInt32}()
for (i,w) in enumerate(words); widx[w] = i; end
cidx = Dict{String,UInt8}()
for (i,c) in enumerate(classes); cidx[c] = i; end

@info "Saving dict.jld2"
save("dict.jld2", "wcnt", d, "words", words, "widx", widx,
                  "ccnt", c, "classes", classes, "cidx", cidx)
