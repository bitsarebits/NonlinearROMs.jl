"""
    MeshGraph{Ti<:Integer,Tv<:Real} <: Graphs.AbstractSimpleGraph{Ti}

A directed graph with the same internal layout as `Graphs.SimpleDiGraph{Ti}`
(a pair of sorted forward/backward adjacency lists), extended with a scalar
weight attached to every directed edge from both endpoints' perspectives:

- `fweights[s][k]` is the weight of the edge `s -> fadjlist[s][k]`, stored
  alongside `fadjlist` for O(1) access when iterating `outneighbors(g,s)`.
- `bweights[d][k]` is the weight of the edge `badjlist[d][k] -> d`, stored
  alongside `badjlist` for O(1) access when iterating `inneighbors(g,d)`.

By default `add_edge!` stores the same value on both sides of an edge; pass
distinct forward/backward weights explicitly for asymmetric couplings (e.g.
upwinded or non-symmetric operators on a mesh).
"""
mutable struct MeshGraph{Ti<:Integer,Tv<:Real} <: Graphs.AbstractSimpleGraph{Ti}
  ne::Int
  fadjlist::Vector{Vector{Ti}}
  badjlist::Vector{Vector{Ti}}
  fweights::Vector{Vector{Tv}}
  bweights::Vector{Vector{Tv}}

  function MeshGraph{Ti,Tv}(
    ne::Int,
    fadjlist::Vector{Vector{Ti}},
    badjlist::Vector{Vector{Ti}},
    fweights::Vector{Vector{Tv}},
    bweights::Vector{Vector{Tv}}
    ) where {Ti<:Integer,Tv<:Real}

    Graphs.SimpleGraphs.throw_if_invalid_eltype(Ti)
    return new{Ti,Tv}(ne,fadjlist,badjlist,fweights,bweights)
  end
end

function MeshGraph(
  ne::Int,
  fadjlist::Vector{Vector{Ti}},
  badjlist::Vector{Vector{Ti}},
  fweights::Vector{Vector{Tv}},
  bweights::Vector{Vector{Tv}}
  ) where {Ti<:Integer,Tv<:Real}

  return MeshGraph{Ti,Tv}(ne,fadjlist,badjlist,fweights,bweights)
end

function MeshGraph{Ti,Tv}(n::Integer=0) where {Ti<:Integer,Tv<:Real}
  fadjlist = [Vector{Ti}() for _ in 1:n]
  badjlist = [Vector{Ti}() for _ in 1:n]
  fweights = [Vector{Tv}() for _ in 1:n]
  bweights = [Vector{Tv}() for _ in 1:n]
  return MeshGraph{Ti,Tv}(0,fadjlist,badjlist,fweights,bweights)
end

MeshGraph(n::Ti) where {Ti<:Integer} = MeshGraph{Ti,Float64}(n)
MeshGraph() = MeshGraph{Int,Float64}()

"""
    MeshGraph(g::SimpleDiGraph, fw::Real=one(Float64), bw::Real=fw)

Build a `MeshGraph` from an existing `SimpleDiGraph`, assigning the
constant weights `fw`/`bw` to every edge.
"""
function MeshGraph(g::SimpleDiGraph{Ti},fw::Tv=1.0,bw::Tv=fw) where {Ti<:Integer,Tv<:Real}
  mg = MeshGraph{Ti,Tv}(nv(g))
  for e in edges(g)
    add_edge!(mg,src(e),dst(e),fw,bw)
  end
  return mg
end

# core AbstractSimpleGraph interface

Graphs.edgetype(::MeshGraph{Ti}) where {Ti} = Edge{Ti}
Graphs.is_directed(::Type{<:MeshGraph}) = true

Graphs.SimpleGraphs.badj(g::MeshGraph) = g.badjlist
Graphs.SimpleGraphs.badj(g::MeshGraph,v::Integer) = g.badjlist[v]

function Base.copy(g::MeshGraph{Ti,Tv}) where {Ti,Tv}
  MeshGraph{Ti,Tv}(
    g.ne,
    Graphs.deepcopy_adjlist(g.fadjlist),
    Graphs.deepcopy_adjlist(g.badjlist),
    Graphs.deepcopy_adjlist(g.fweights),
    Graphs.deepcopy_adjlist(g.bweights)
  )
end

function Base.:(==)(g::MeshGraph,h::MeshGraph)
  (
    vertices(g) == vertices(h) &&
    ne(g) == ne(h) &&
    g.fadjlist == h.fadjlist &&
    g.badjlist == h.badjlist &&
    g.fweights == h.fweights &&
    g.bweights == h.bweights
  )
end

function Graphs.has_edge(g::MeshGraph{Ti},e::Edge{Ti}) where {Ti}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false
  @inbounds list = g.fadjlist[s]
  @inbounds list_backedge = g.badjlist[d]
  if length(list) > length(list_backedge)
    d = s
    list = list_backedge
  end
  return Graphs.insorted(d,list)
end

# weighted edge insertion/removal

"""
    add_edge!(g::MeshGraph, e, fw, bw=fw)
    add_edge!(g::MeshGraph, s, d, fw, bw=fw)

Insert the directed edge `s -> d` with forward weight `fw` (cached on the
`s`-side, next to `fadjlist[s]`) and backward weight `bw` (cached on the
`d`-side, next to `badjlist[d]`). Returns `false` (and does not overwrite
existing weights) if the edge is already present.
"""
function Graphs.add_edge!(g::MeshGraph{Ti,Tv},e::Edge{Ti},fw::Tv,bw::Tv=fw) where {Ti,Tv}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false

  @inbounds flist = g.fadjlist[s]
  findex = searchsortedfirst(flist,d)
  @inbounds (findex <= length(flist) && flist[findex] == d) && return false
  insert!(flist,findex,d)
  insert!(g.fweights[s],findex,fw)

  g.ne += 1

  @inbounds blist = g.badjlist[d]
  bindex = searchsortedfirst(blist,s)
  insert!(blist,bindex,s)
  insert!(g.bweights[d],bindex,bw)

  return true
end

function Graphs.add_edge!(g::MeshGraph{Ti,Tv},e::Edge{Ti}) where {Ti,Tv}
  return add_edge!(g,e,one(Tv))
end

function Graphs.add_edge!(g::MeshGraph{Ti,Tv},s::Integer,d::Integer,fw::Tv,bw::Tv=fw) where {Ti,Tv}
  return add_edge!(g,Edge{Ti}(Ti(s),Ti(d)),fw,bw)
end

function Graphs.rem_edge!(g::MeshGraph{Ti},e::Edge{Ti}) where {Ti}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false

  @inbounds flist = g.fadjlist[s]
  findex = searchsortedfirst(flist,d)
  @inbounds (findex <= length(flist) && flist[findex] == d) || return false
  deleteat!(flist,findex)
  deleteat!(g.fweights[s],findex)

  g.ne -= 1

  @inbounds blist = g.badjlist[d]
  bindex = searchsortedfirst(blist,s)
  deleteat!(blist,bindex)
  deleteat!(g.bweights[d],bindex)

  return true
end

function Graphs.add_vertex!(g::MeshGraph{Ti,Tv}) where {Ti,Tv}
  (nv(g) + one(Ti) <= nv(g)) && return false
  push!(g.fadjlist,Vector{Ti}())
  push!(g.badjlist,Vector{Ti}())
  push!(g.fweights,Vector{Tv}())
  push!(g.bweights,Vector{Tv}())
  return true
end

# weight accessors

"""
    get_weight(g::MeshGraph, s, d) -> Real

Forward weight of the edge `s -> d` (i.e. the value stored in
`fweights[s]`), or `zero(Tv)` if the edge does not exist.
"""
function get_weight(g::MeshGraph{Ti,Tv},s::Integer,d::Integer) where {Ti,Tv}
  verts = vertices(g)
  (s in verts && d in verts) || return zero(Tv)
  @inbounds list = g.fadjlist[s]
  index = searchsortedfirst(list,d)
  @inbounds (index <= length(list) && list[index] == d) || return zero(Tv)
  return @inbounds g.fweights[s][index]
end

get_weight(g::MeshGraph,e::Edge) = get_weight(g,src(e),dst(e))

"""
    out_weights(g::MeshGraph, v) -> Vector

Forward weights of the outgoing edges of `v`, in the same order as
`outneighbors(g,v)`.
"""
out_weights(g::MeshGraph,v::Integer) = g.fweights[v]

"""
    in_weights(g::MeshGraph, v) -> Vector

Backward weights of the incoming edges of `v`, in the same order as
`inneighbors(g,v)`.
"""
in_weights(g::MeshGraph,v::Integer) = g.bweights[v]

"""
    Graphs.weights(g::MeshGraph) -> SparseMatrixCSC

The forward-weight adjacency matrix `W`, with `W[s,d]` equal to the forward
weight of edge `s -> d` (zero where no edge exists), suitable for use as a
`distmx` argument to Graphs.jl's shortest-path algorithms.
"""
function Graphs.weights(g::MeshGraph{Ti,Tv}) where {Ti,Tv}
  n = nv(g)
  I = Ti[]
  J = Ti[]
  V = Tv[]
  for s in vertices(g)
    for (k,d) in enumerate(g.fadjlist[s])
      push!(I,s)
      push!(J,d)
      push!(V,g.fweights[s][k])
    end
  end
  return sparse(I,J,V,n,n)
end