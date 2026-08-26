"""
    struct DeepONet{F} <: NeuralNetwork
      branch_layers::Tuple{Vararg{Int}}
      trunk_layers::Tuple{Vararg{Int}}
      activation::F
    end

Explicit architectural configuration for a Deep Operator Network (DeepONet).
A DeepONet consists of two sub-networks:
1. **Branch Net**: Processes the input parameters/sensors \$u\$.
2. **Trunk Net**: Processes the spatial/spatiotemporal continuous coordinates \$y\$.

The final output is computed via the dot product of their feature vectors:
\$\$G(u)(y) = \\sum_{k=1}^{p} b_k(u) t_k(y)\$\$

The last layer of both `branch_layers` and `trunk_layers` must have the exact same dimension \$p\$.

# Examples
The layers are defined as standard Julia tuples.
```julia
# Branch Net: 2 inputs -> 64 hidden -> 32 output (latent dimension p=32)
# Trunk Net: 3 inputs (e.g., 2D space + time) -> 64 hidden -> 32 output
model = DeepONet(
  branch_layers = (2, 64, 32),
  trunk_layers  = (3, 64, 32),
  activation    = relu
)
```
"""
struct DeepONet{F} <: NeuralNetwork
  branch_layers::Tuple{Vararg{Int}}
  trunk_layers::Tuple{Vararg{Int}}
  activation::F
end

function DeepONet(;branch_layers,trunk_layers,activation=tanh)
  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end

function DeepONet(
  nbranch_in::Int,
  ntrunk_in::Int;
  width::Int=64,
  depth::Int=3,
  hidden = ntuple(_ -> width,depth),
  branch_layers=(nbranch_in,hidden...,width),
  trunk_layers=(ntrunk_in,hidden...,width),
  activation=tanh
  )

  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end

"""
    struct NOMAD{F} <: NeuralNetwork
      approximator_layers::Tuple{Vararg{Int}}
      decoder_layers::Tuple{Vararg{Int}}
      activation::F
    end

Explicit architectural configuration for a NOMAD (Non-linear Manifold Decoder) network.
It uses an Approximator (Encoder) to map the parametric sensors into a latent space, and a Decoder that takes the concatenated vector of the latent representation and the physical coordinates to predict the solution field.

# Examples
The layers are defined as standard Julia tuples.
```julia
# Approximator: 5 sensors -> 32 hidden -> 16 latent space
# Decoder: 19 inputs (16 latent + 3 spatial coords) -> 32 hidden -> 1 output
model = NOMAD(
  approximator_layers = (5, 32, 16),
  decoder_layers      = (19, 32, 1),
  activation          = relu
)
```
"""
struct NOMAD{F} <: NeuralNetwork
  approximator_layers::Tuple{Vararg{Int}}
  decoder_layers::Tuple{Vararg{Int}}
  activation::F
end

function NOMAD(;approximator_layers,decoder_layers,activation=tanh)
  NOMAD(Tuple(approximator_layers),Tuple(decoder_layers),activation)
end

function NOMAD(
  nsensors_in::Int,
  ncoords_in::Int;
  width::Int=64,
  depth::Int=3,
  hidden = ntuple(_ -> width,depth),
  approximator_layers=(nsensors_in,hidden...,width),
  decoder_layers=(width+ncoords_in,hidden...,1),
  activation=tanh
  )

  NOMAD(Tuple(approximator_layers),Tuple(decoder_layers),activation)
end
