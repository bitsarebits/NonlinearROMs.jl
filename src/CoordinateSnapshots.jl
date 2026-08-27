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

# """
#     coords_matrix(V::SingleFieldFESpace) -> Matrix{Float32}

# Full-resolution counterpart of `get_formatted_data`'s coordinate stacking: stacks
# every DoF coordinate of `V` (no spatial subsampling) into a `(D_phys,N_dofs)` matrix.
# Used at inference time, where predictions are required at every DoF regardless of the
# spatial subsampling used during training.
# """
# coords_matrix(V::SingleFieldFESpace) = Float32.(stack(p -> collect(p.data),vec(get_coords(V))))

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

struct InputData{A<:AbstractRealisation,Tc,Nc,B<:AbstractArray{Tc,Nc}}
  r::A
  coords::B
end

function InputData(r::AbstractRealisation,V::FESpace)
  coords = get_coords(V)
  InputData(r,coords)
end

ParamDataStructures.get_realisation(s::InputData) = s.r
get_coords(s::InputData) = s.coords

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

"""
    _spacetime_coords(coords_raw::AbstractMatrix,t_grid::AbstractVector) -> Matrix

Builds the flattened `(D_phys+1,N_dofs*N_time)` space-time trunk-input grid used by
transient DeepONet/NOMAD: for every time value (outer loop) and every spatial DoF
(inner loop), appends the time as an extra last coordinate. Space varies fastest, so
the column order matches how `get_formatted_data`/`_flatten` flatten the target data.
"""
function _spacetime_coords(coords_raw::AbstractMatrix{T},t_grid::AbstractVector{T}) where T
  D_phys,N_dofs = size(coords_raw)
  N_time = length(t_grid)
  coords = zeros(T,D_phys+1,N_dofs*N_time)
  col = 1
  @views for t_val in t_grid
    for x_idx in 1:N_dofs
      coords[1:D_phys,col] .= coords_raw[:,x_idx]
      coords[D_phys+1,col] = t_val
      col += 1
    end
  end
  return coords
end

function get_formatted_data(::Type{T},s::TransientCoordinateSnapshots) where T
  data_3d,params = get_formatted_data(T,s.snaps) # data_3d: (N_dofs,n_samples,N_time)
  t_grid = T.(get_times(get_realisation(s)))
  coords_raw = T.(stack(p -> collect(p.data),vec(get_coords(s)))) # (D_phys,N_dofs)
  coords = _spacetime_coords(coords_raw,t_grid)

  N_dofs,n_samples,N_time = size(data_3d)
  data = zeros(T,N_dofs*N_time,n_samples)
  for i in 1:n_samples
    col = 1
    for t_idx in 1:N_time, x_idx in 1:N_dofs
      data[col,i] = data_3d[x_idx,i,t_idx]
      col += 1
    end
  end

  return data,params,coords
end

function get_formatted_data(::Type{T},s::InputData) where T
  params = T.(matrix_of_params(get_realisation(s)))
  coords = T.(stack(p -> collect(p.data),vec(get_coords(s))))
  return (params,coords)
end

function get_formatted_data(::Type{T},s::InputData{<:TransientRealisation}) where T
  r = get_realisation(s)
  params = T.(matrix_of_params(r))
  t_grid = T.(get_times(r))
  coords_raw = T.(stack(p -> collect(p.data),vec(get_coords(s)))) # (D_phys,N_dofs)
  coords = _spacetime_coords(coords_raw,t_grid)
  return (params,coords)
end

function get_formatted_data(s)
  get_formatted_data(Float32,s)
end