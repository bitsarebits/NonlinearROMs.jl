struct Sampler{A}
  strategy::A
end

Sampler(s::Sampler) = s
Sampler() = Sampler(identity)
NoSampler() = Sampler(nothing)

sample(args...) = @notimplemented
sample(s::Sampler,x::AbstractArray,args...) = @abstractmethod

sample(s::Sampler{typeof(identity)},x::AbstractArray,args...) = x

function sample(s::Sampler{<:Function},x::AbstractArray,axis=1)
  y = selectdim(x,axis,1)
  z = s.strategy(y)
  sx = similar(z,(size(z)...,size(x,axis)))
  @views for i in 1:size(x,axis)
    y = selectdim(x,axis,i)
    z = s.strategy(y)
    selectdim(sx,axis,i) .= z
  end
  return sx
end

function sample(s::Sampler{<:Integer},x::AbstractArray,axis=1)
  step = s.strategy
  selectdim(x,axis,1:step:size(x,axis))
end

function sample(s::Sampler,x::BlockSnapshots,axis=1)
  @notimplemented "Do this!"
end

# coordinate snapshots sampling

for T in (:(typeof(identity)),:Function,:Integer)
  @eval begin
    function sample(s::Sampler{<:$T},x::CoordinateSnapshots,axis=1)
      sx = sample(s,x.snaps,axis)
      xx = sample(s,get_coords(x))
      CoordinateSnapshots(sx,xx)
    end
  end
end

struct NeuralSampler{A,B,C}
  space_sampler::Sampler{A}
  param_sampler::Sampler{B}
  time_sampler::Sampler{C}
end

function NeuralSampler(s,p,t)
  space_sampler = Sampler(s)
  param_sampler = Sampler(p)
  time_sampler = Sampler(t)
  NeuralSampler(space_sampler,param_sampler,time_sampler)
end

function NeuralSampler(;space_step=1,param_step=1,time_step=nothing)
  NeuralSampler(space_step,param_step,time_step)
end

function sample(s::NeuralSampler,x::AbstractArray{<:Point})
  points = sample(s.space_sampler,vec(x))
  Float32.(stack(p -> collect(p.data),vec(points)))
end

function param_sample(s::NeuralSampler{A,Nothing},x::Snapshots) where A
  @notimplemented
end

function param_sample(s::NeuralSampler{A,typeof(identity)},x::Snapshots) where A
  x
end

function param_sample(s::NeuralSampler,x::Snapshots)
  select_snapshots(x,get_param_ids(s.param_sampler,x))
end

function time_sample(s::NeuralSampler{A,Nothing},x::TransientSnapshots) where A
  @notimplemented
end

function time_sample(s::NeuralSampler{A,typeof(identity)},x::TransientSnapshots) where A
  x
end

function time_sample(s::NeuralSampler,x::TransientSnapshots)
  select_times(x,get_time_ids(s.time_sampler,x))
end

const SteadyNeuralSampler{A,B} = NeuralSampler{A,B,Nothing}

function sample(s::SteadyNeuralSampler,x::SteadySnapshots)
  sp = param_sample(s,x)
  space_axis = 1
  sample(sp,x,space_axis)
end

const TransientNeuralSampler{A,B,C} = NeuralSampler{A,B,C}

function sample(s::TransientNeuralSampler,x::TransientSnapshots)
  xp = param_sample(s,x)
  xpt = time_sample(s,xp)
  space_axis = 1
  sample(xpt,x,space_axis)
end

get_space_ids(s::NeuralSampler,args...) = get_ids(s.space_sampler,args...)
get_param_ids(s::NeuralSampler,args...) = get_ids(s.param_sampler,args...)
get_time_ids(s::NeuralSampler,args...) = get_ids(s.time_sampler,args...)

# utils

"""
    get_ids(s::Sampler,n::Int) -> AbstractVector{Int}

Resolves the explicit set of indices (out of `1:n`) selected by `s`: the full
range strided by `s.strategy` when it is an `Integer`, `s.strategy` itself when
it is already an `AbstractVector` of indices, or `1:n` (no subsampling) when
`s.strategy` is `nothing`.
"""
get_ids(s::Sampler{<:AbstractVector},n::Int) = s.strategy
get_ids(s::Sampler{<:Integer},n::Int) = 1:s.strategy:n
get_ids(s::Sampler{Nothing},n::Int) = 1:n

for (f,g) in zip((:get_param_ids,:get_time_ids),(:num_params,:num_times))
  @eval begin
    function $f(s::Sampler{<:AbstractVector},x::Snapshots)
      return s.strategy
    end

    function $f(s::Sampler{<:Integer},x::Snapshots)
      step = s.strategy
      ids = 1:step:$g(x)
      return ids
    end
  end
end