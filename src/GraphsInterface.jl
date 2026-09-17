function build_graph!(g::MeshBasedGraph,cell_to_dofs::Table,dofs_to_coords::AbstractVector{<:Point})
  dof_to_cells = inverse_table(cell_to_dofs)
  cc = array_cache(cell_to_dofs)
  cd = array_cache(dof_to_cells)
  for dof in eachindex(dof_to_cells)
    cells = getindex!(cd,dof_to_cells,dof)
    for cell in cells
      dofs = getindex!(cc,cell_to_dofs,cell)
      for neighbor in dofs
        w = norm(dofs_to_coords[dof] - dofs_to_coords[neighbor])
        add_edge!(g,dof,neighbor,w)
      end
    end
  end
  g
end