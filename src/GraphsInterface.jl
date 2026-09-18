include("WeightedSimpleDiGraphs.jl")

abstract type GraphStrategy end

build_graph(s::GraphStrategy,V::FESpace) = @abstractmethod

struct MeshGraph <: GraphStrategy end

function build_graph(s::MeshGraph,V::FESpace)
  cell_to_dofs = Table(get_cell_dof_ids(V))
  dof_to_coords = get_free_dof_coordinates(V)
  g = WeightedSimpleDiGraph(length(dof_to_coords))
  build_graph!(g,s,cell_to_dofs,dof_to_coords)
end

function build_graph!(
  g::WeightedSimpleDiGraph,
  s::MeshGraph,
  cell_to_dofs::Table,
  dof_to_coords::AbstractVector{<:Point}
  )

  dof_to_cells = inverse_table(cell_to_dofs)
  cc = array_cache(cell_to_dofs)
  cd = array_cache(dof_to_cells)
  for dof in eachindex(dof_to_cells)
    cells = getindex!(cd,dof_to_cells,dof)
    for cell in cells
      dofs = getindex!(cc,cell_to_dofs,cell)
      for neighbor in dofs
        neighbor <= 0 && continue
        w = norm(dof_to_coords[dof] - dof_to_coords[neighbor])
        add_edge!(g,dof,neighbor,w)
      end
    end
  end
  g
end

struct DistanceGraph <: GraphStrategy
  k::Int
  max_distance::Real
end

function build_graph(s::DistanceGraph,V::FESpace)
  dof_to_coords = get_free_dof_coordinates(V)
  data = map(x -> SVector(Tuple(x)),dof_to_coords)
  D = num_cell_dims(get_triangulation(V))
  metric = Minkowski(D)
  kdtree = KDTree(data,metric)
  g = WeightedSimpleDiGraph(length(dof_to_coords))
  build_graph!(g,s,kdtree,dof_to_coords)
end

function build_graph!(
  g::WeightedSimpleDiGraph,
  s::DistanceGraph,
  kdtree::NNTree,
  dof_to_coords::AbstractVector{<:Point}
  )

  for (dof,coord) in enumerate(dof_to_coords)
    neighbors,distances = search(s,kdtree,coord)
    for (neighbor,w) in zip(neighbors,distances)
      add_edge!(g,dof,neighbor,w)
    end
  end
  g
end

function search(strategy::DistanceGraph,kdtree::NNTree,x::Point)
  x′ = get_array(ForwardDiff.value(x))
  v,d = knn(kdtree,x′,strategy.k,true)
  nkeep = 0
  for di in d
    if di <= strategy.max_distance
      nkeep += 1
    end
  end
  vk = zeros(eltype(v),nkeep)
  dk = zeros(eltype(d),nkeep)
  nkeep = 0
  for (vi,di) in zip(v,d)
    if di <= strategy.max_distance
      nkeep += 1
      vk[nkeep] = vi
      dk[nkeep] = di
    end
  end
  vk,dk
end