using Pkg
for p in ("Knet","CSV","FileIO","JLD2","WordTokenizers","IterTools")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end
