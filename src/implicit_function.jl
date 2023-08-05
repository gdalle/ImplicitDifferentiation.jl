"""
    ImplicitFunction{F,C,L,B}

Differentiable wrapper for an implicit function defined by a forward mapping `y` and a set of conditions `c`.

The Jacobian of `y(⋅)` is computed using the implicit function theorem:

    ∂/∂y c(x, y(x)) * ∂y(x) = -∂/∂x c(x, y(x))

This requires solving a linear system `A * J = -B`.

# Constructors

You can construct an `ImplicitFunction` from two callables (function-like objects) `forward` and `conditions`.

    ImplicitFunction(
        forward, conditions;
        linear_solver=IterativeLinearSolver(), conditions_backend=nothing,
    )

While `forward` does not not need to be compatible with automatic differentiation, `conditions` has to be (with the provided `conditions_backend` if there is one).

There are two possible signatures for `forward` and `conditions`, which must be consistent with one another:
    
    1. Standard: `forward(x; kwargs...) = y` and `conditions(x, y; kwargs...) = c`
    2. Byproduct: `forward(x; kwargs...) = (y, z)` and `conditions(x, y, z; kwargs...) = c`.
    
In both cases, `x`, `y` and `c` must be arrays with `size(y) = size(c)`.
In the second case, the byproduct `z` can be an arbitrary object generated by `forward`, but beware that we consider it constant for differentiation purposes.

# Callable behavior

An `ImplicitFunction` object `implicit` behaves like a function, and every call is differentiable.
    
    implicit(x::AbstractArray; kwargs...)

This returns exactly `implicit.forward(x; kwargs...)`, which as we mentioned can be either an array `y` or a tuple `(y, z)`.

# Fields

- `forward::F`
- `conditions::C`
- `linear_solver::L<:AbstractLinearSolver`
- `conditions_backend::B<:Union{Nothing,AbstractBackend}`

!!! warning "Warning"
    At the moment, `conditions_backend` can only be `nothing` or `AD.ForwardDiffBackend()`. We are investigating why the other backends fail.
"""
struct ImplicitFunction{F,C,L<:AbstractLinearSolver,B<:Union{Nothing,AbstractBackend}}
    forward::F
    conditions::C
    linear_solver::L
    conditions_backend::B

    function ImplicitFunction(
        forward::F,
        conditions::C;
        linear_solver::L=IterativeLinearSolver(),
        conditions_backend::B=nothing,
    ) where {F,C,L,B}
        return new{F,C,L,B}(forward, conditions, linear_solver, conditions_backend)
    end
end

function Base.show(io::IO, implicit::ImplicitFunction)
    @unpack forward, conditions, linear_solver, conditions_backend = implicit
    return print(
        io, "ImplicitFunction($forward, $conditions, $linear_solver, $conditions_backend)"
    )
end

function (implicit::ImplicitFunction)(x::AbstractArray; kwargs...)
    y_or_yz = implicit.forward(x; kwargs...)
    if !(
        y_or_yz isa AbstractArray ||  # 
        (y_or_yz isa Tuple && length(y_or_yz) == 2 && y_or_yz[1] isa AbstractArray)
    )
        throw(
            DimensionMismatch(
                "The forward mapping must return an array `y` or a tuple `(y, z)`"
            ),
        )
    end
    return y_or_yz
end
