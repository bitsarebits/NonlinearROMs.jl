using Documenter
using NonlinearROMs

makedocs(;
  modules=[NonlinearROMs],
  format=Documenter.HTML(size_threshold=nothing),
  pages=[
    "Home" => "index.md",
    "Description" => [
      "Neural Operators" => "neurals.md",
    ],
  ],
  sitename="NonlinearROMs.jl",
  warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo="github.com/nichomueller/NonlinearROMs.jl.git",
  push_preview=true,
)
