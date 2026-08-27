"""
    get_coords(V::SingleFieldFESpace) -> Array{Point{D,Float64}}

Extracts the physical coordinates of the free DoFs of `V`, indexed consistently
with `V`'s own free-dof numbering (the same numbering used by
`get_all_data`/`get_free_dof_values` on `V`). Since the mapping is obtained by
directly interpolating the coordinate field onto `V`, it is valid for any
`SingleFieldFESpace` -- no special DoF ordering is required. Use
`sample(sampler::NeuralSampler,get_coords(V))` to subsample and stack the
result into a `(D_phys,N_dofs)` `Matrix{Float32}`.
"""
function get_coords(V::SingleFieldFESpace)
  order = get_polynomial_orders(V)
  trian = get_triangulation(V)
  model = get_background_model(trian)
  get_coords(model,order)
end

function get_coords(model::CartesianDiscreteModel{D},orders::NTuple{D,Int}) where D 
  desc = get_cartesian_descriptor(model)
  cells = CartesianIndices(desc.partition)
  nodes = CartesianIndices(orders .* desc.partition .+ 1 .- desc.isperiodic)
  coords = Array{Point{D,Float64}}(undef,size(nodes))
  for cell in cells
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    nodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:(ni+orders[i])
    end
    for inode in Iterators.product(nodes_range...)
      _is_periodic_node(inode,nodes) && continue
      coords[inode...] = Point(ntuple(d -> desc.origin[d] + (inode[d]-1)*desc.sizes[d],Val{D}()))
    end
  end
  return coords
end

function _is_periodic_node(inode,nodes)
  try
    nodes[inode...]
    return false
  catch
    return true
  end
end

struct CoordinateSnapshots{T,N,Tc,Nc,A<:AbstractSnapshots{T,N},B<:AbstractArray{Tc,Nc}} <:AbstractSnapshots{T,N}
  snaps::A
  coords::B
end

function CoordinateSnapshots(snaps::AbstractSnapshots,V::FESpace)
  coords = get_coords(V)
  CoordinateSnapshots(snaps,coords)
end

const SteadyCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:SteadySnapshots{T,N}}
const TransientCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:TransientSnapshots{T,N}}

ParamDataStructures.get_all_data(s::CoordinateSnapshots) = get_all_data(s.snaps)
ParamDataStructures.get_param_data(s::CoordinateSnapshots) = get_param_data(s.snaps)
ParamDataStructures.get_initial_param_data(s::CoordinateSnapshots) = get_initial_param_data(s.snaps)
DofMaps.get_dof_map(s::CoordinateSnapshots) = get_dof_map(s.snaps)
ParamDataStructures.get_realisation(s::CoordinateSnapshots) = get_realisation(s.snaps)
get_coords(s::CoordinateSnapshots) = s.coords

function ParamDataStructures.select_snapshots(s::CoordinateSnapshots,pindex) 
  snaps = select_snapshots(s.snaps,pindex)
  CoordinateSnapshots(snaps,s.coords)
end

function ParamDataStructures.select_times(s::CoordinateSnapshots,tindex) 
  snaps = select_times(s.snaps,tindex)
  CoordinateSnapshots(snaps,s.coords)
end

function Base.getindex(s::CoordinateSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(s.snaps,i...)
end

function Base.setindex!(s::CoordinateSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  setindex!(s.snaps,v,i...)
end

function get_formatted_data(::Type{T},s::AbstractSnapshots) where T
  data = T.(get_all_data(s))
  params = T.(matrix_of_params(get_realisation(s)))
  return (data,params)
end

function get_formatted_data(::Type{T},s::CoordinateSnapshots) where T
  data,params = get_formatted_data(T,s.snaps)
  coords = T.(stack(p -> collect(p.data),vec(get_coords(s))))
  return (data,params,coords)
end

function get_formatted_data(s::AbstractSnapshots)
  get_formatted_data(Float32,s)
end