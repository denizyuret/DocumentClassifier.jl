using FileIO, CSV, WordTokenizers
classes = ["entertainment", "health", "sports", "us", "world", "technology", "politics", "science", "business"];
set_tokenizer(poormans_tokenize) # removes all punctuation, splits on space
f = CSV.File("ML_Dataset_20190227.csv")
d = Dict{String,UInt32}()
for r in f, w in tokenize(r.text) # should we lowercase?
    d[w] = 1 + get(d,w,0)
end
a = sort(collect(keys(d)), by=(x->d[x]), rev=true)
id = Dict{String,UInt32}()
for (i,w) in enumerate(a)
    id[w] = i
end
save("dict.jld2", "count", d, "words", a, "w2i", id, "classes", classes)
