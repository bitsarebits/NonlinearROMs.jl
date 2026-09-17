# loggers 

mutable struct TrainingLog
  name::String
  max_epochs::Int
  print_every::Int
  verbose::Bool
  t_start::Float64
  t_start_fast::Float64
end

function TrainingLog(name::String,max_epochs::Int;verbose::Bool=true,print_every=500)
  TrainingLog(name,max_epochs,print_every,verbose,0.0,0.0)
end

function init!(log::TrainingLog)
  !log.verbose && return nothing

  log.t_start = time()
  @info "Starting $(log.name) Training on Reactant Device (First epoch compiles XLA...)"
  return nothing
end

function update!(log::TrainingLog,epoch::Int,current_loss::Real)
  !log.verbose && return nothing

  if epoch == 1
    log.t_start_fast = time()
    comp_mins = round((log.t_start_fast - log.t_start) / 60,digits=2)
    @info "Compilation finished in $comp_mins min. Fast training started."
  end

  if epoch == 1 || epoch % log.print_every == 0 || epoch == log.max_epochs
    elapsed_fast = time() - log.t_start_fast
    time_per_epoch = epoch > 1 ? elapsed_fast / (epoch - 1) : 0.0
    eta_seconds = time_per_epoch * (log.max_epochs - epoch)

    msg = "> Epoch: $(lpad(epoch,5)) \t Loss: $(Float32(current_loss)) \t ETA: $(format_eta(eta_seconds))"
    println(msg)
  end
  return nothing
end

function finalize!(log::TrainingLog)
  !log.verbose && return nothing

  total_mins = round((time() - log.t_start) / 60,digits=2)
  @info "Training $(log.name) Completed in $total_mins minutes"
  return nothing
end

function format_eta(eta_seconds::Real)
  eta_sec = round(Int,eta_seconds)
  h = div(eta_sec,3600)
  m = div(rem(eta_sec,3600),60)
  s = rem(eta_sec,60)
  return h > 0 ? "$(lpad(h,2,'0')):$(lpad(m,2,'0')):$(lpad(s,2,'0'))" :
         "$(lpad(m,2,'0')):$(lpad(s,2,'0'))"
end

# normalisation handling 

struct ZscoreStats{A<:AbstractVector,B<:AbstractVector}
  μ::A
  σ::B
end

function ZscoreStats(data::AbstractMatrix;normalise=false)
  if normalise
    stats = ZscoreStats(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=2),dims=2)
  σ = dropdims(std(data,dims=2),dims=2)
  # Avoid dividing by zero if a feature is constant
  for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return ZscoreStats(μ,σ)
end

struct NormStats{T<:Real,A<:ZscoreStats,B<:ZscoreStats}
  dmax::T
  pscore::A 
  xscore::B
end

function NormStats(data,params,coords;normalise=false)
  dmax = maximum(abs,data)
  normalise && (data ./= dmax)
  input = ZscoreStats(params;normalise)
  output = ZscoreStats(coords;normalise)
  NormStats(dmax,input,output)
end

normalise!(args...) = @abstractmethod

function normalise!(data::AbstractVector,stats::ZscoreStats)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function normalise!(data::AbstractMatrix,stats::ZscoreStats)
  for v in eachcol(data)
    normalise!(v,stats)
  end
  data
end

function normalise!(inout::NTuple{2,AbstractArray},stats::NormStats)
  a,b = inout
  normalise!(a,stats.pscore)
  normalise!(b,stats.xscore)
end

function normalise!(inout::NTuple{3,AbstractArray},stats::NormStats)
  a,b,c = inout
  a ./= stats.dmax
  normalise!(b,stats.pscore)
  normalise!(c,stats.xscore)
end

# Data types

function FESpaces.get_free_dof_coordinates(V::MultiFieldFESpace)
  map(get_free_dof_coordinates,V.spaces)
end

struct CoordinateSnapshots{T,N,Tc,Nc,A<:AbstractSnapshots{T,N},B<:AbstractArray{Tc,Nc}} <:AbstractSnapshots{T,N}
  snaps::A
  coords::B
end

function CoordinateSnapshots(snaps::AbstractSnapshots,V::FESpace)
  coords = get_free_dof_coordinates(V)
  CoordinateSnapshots(snaps,coords)
end

const SteadyCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:SteadySnapshots{T,N}}
const TransientCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:TransientSnapshots{T,N}}

ParamDataStructures.get_all_data(s::CoordinateSnapshots) = get_all_data(s.snaps)
ParamDataStructures.get_param_data(s::CoordinateSnapshots) = get_param_data(s.snaps)
ParamDataStructures.get_initial_param_data(s::CoordinateSnapshots) = get_initial_param_data(s.snaps)
DofMaps.get_dof_map(s::CoordinateSnapshots) = get_dof_map(s.snaps)
ParamDataStructures.get_realisation(s::CoordinateSnapshots) = get_realisation(s.snaps)
get_coordinates(s::CoordinateSnapshots) = s.coords

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
  coords = T.(stack(p -> collect(p.data),vec(get_coordinates(s))))
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
  coords_raw = T.(stack(p -> collect(p.data),vec(get_coordinates(s)))) # (D_phys,N_dofs)
  coords = _spacetime_coords(coords_raw,t_grid)

  N_dofs,n_samples,N_time = size(data_3d)
  data = zeros(T,N_dofs*N_time,n_samples)
  for i in 1:n_samples
    col = 1
    for t_idx in 1:N_time,x_idx in 1:N_dofs
      data[col,i] = data_3d[x_idx,i,t_idx]
      col += 1
    end
  end

  return data,params,coords
end

function get_formatted_data(::Type{T},r::AbstractRealisation,coords::AbstractArray{<:Point}) where T
  params = T.(matrix_of_params(r))
  coords_mat = T.(stack(p -> collect(p.data),vec(coords)))
  return (params,coords_mat)
end

function get_formatted_data(::Type{T},r::TransientRealisation,coords::AbstractArray{<:Point}) where T
  params = T.(matrix_of_params(r))
  t_grid = T.(get_times(r))
  coords_raw = T.(stack(p -> collect(p.data),vec(coords))) # (D_phys,N_dofs)
  coords_mat = _spacetime_coords(coords_raw,t_grid)
  return (params,coords_mat)
end

function get_formatted_data(s)
  get_formatted_data(Float32,s)
end

# Constructs the 3D input tensor required by Kernel Neural Operators.
# It concatenates the physical coordinates and parameter values to form the 
# vector field (x, a(x)) representing the geometry and input function.
# Returns a 3D tensor of size (dim_params + dim_x, n_nodes, n_samples) ready for the Lifting layer.
function _build_kernel_inputs(params::AbstractArray{T,2}, coords::AbstractArray{T,2}) where T
    dim_params,n_samples = size(params)
    dim_x,n_nodes = size(coords)
    
    # Preallocate the 3D tensor: [Features, Nodes, Samples]
    input_tensor = zeros(T,dim_params + dim_x,n_nodes,n_samples)
    
    @views for s in 1:n_samples
        pₛ = params[:,s]
        for n in 1:n_nodes
            # Concatenate input parameters a(x) and spatial coords x
            input_tensor[1:dim_params,n,s] .= pₛ
            input_tensor[dim_params+1:end,n,s] .= coords[:,n]
        end
    end
    
    return input_tensor
end

# utils 

get_dof_to_nodes(b) = @abstractmethod
get_dof_to_nodes(b::LagrangianDofBasis) = b.nodes[b.dof_to_node]