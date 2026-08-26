abstract type AbstractDeepONet end
abstract type AbstractNOMAD end

"""
    struct DeepONet{F} <: AbstractDeepONet
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
struct DeepONet{F} <: AbstractDeepONet
  branch_layers::Tuple{Vararg{Int}}
  trunk_layers::Tuple{Vararg{Int}}
  activation::F
end

function DeepONet(;branch_layers,trunk_layers,activation=tanh)
  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end

"""
    Base.@kwdef struct AutoDeepONet{F} <: AbstractDeepONet
      width::Int = 64
      depth::Int = 3
      activation::F = tanh
    end

Automatic builder for DeepONet architectures.
It automatically infers the correct input dimensions for the Branch and Trunk networks based on the problem's physical dimensions and the parameter space.

# Fields
- `width::Int`: The number of neurons in each hidden layer, as well as the dimension of the final latent output \$p\$ (default: 64).
- `depth::Int`: The number of hidden layers for both the Branch and Trunk networks (default: 3).
- `activation::F`: The activation function applied to hidden layers (default: `tanh`).

# Examples

```julia
using Lux

# Default architecture
model = AutoDeepONet()

# Custom hyperparameters via keyword arguments
model = AutoDeepONet(width = 128, depth = 4, activation = Lux.gelu)
```
"""
Base.@kwdef struct AutoDeepONet{F} <: AbstractDeepONet
  width::Int = 64
  depth::Int = 3
  activation::F = tanh
end

"""
    struct NOMAD{F} <: AbstractNOMAD
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
struct NOMAD{F} <: AbstractNOMAD
  approximator_layers::Tuple{Vararg{Int}}
  decoder_layers::Tuple{Vararg{Int}}
  activation::F
end

function NOMAD(; approximator_layers, decoder_layers, activation=tanh)
  NOMAD(Tuple(approximator_layers), Tuple(decoder_layers), activation)
end

"""
    Base.@kwdef struct AutoNOMAD{F} <: AbstractNOMAD
      width::Int = 64
      depth::Int = 3
      activation::F = tanh
    end

Automatic builder for NOMAD architectures.
Automatically infers the correct input dimensions for sensors and coordinates based on the physical problem.

# Fields
- `width::Int`: The dimension of the latent space and the number of neurons in the hidden layers (default: 64).
- `depth::Int`: The number of hidden layers for both the Approximator and the Decoder (default: 3).
- `activation::F`: The activation function applied to hidden layers (default: tanh).

# Examples

```julia
using Lux

# Default architecture
model = AutoNOMAD()

# Custom hyperparameters via keyword arguments
model = AutoNOMAD(width = 128, depth = 4, activation = Lux.gelu)
```
"""
Base.@kwdef struct AutoNOMAD{F} <: AbstractNOMAD
  width::Int = 64
  depth::Int = 3
  activation::F = tanh
end
