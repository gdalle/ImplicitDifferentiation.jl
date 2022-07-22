"""
    ImplicitFunction{F,C,L}

Differentiable wrapper for an implicit function `x -> ŷ(x)` whose output is defined by explicit conditions `F(x,ŷ(x)) = 0`.

If `x ∈ ℝⁿ` and `y ∈ ℝᵈ`, then we need as many conditions as output dimensions: `F(x,y) ∈ ℝᵈ`.
Thanks to these conditions, we can compute the Jacobian of `ŷ(⋅)` using the implicit function theorem:
```
∂₂F(x,ŷ(x)) * ∂ŷ(x) = -∂₁F(x,ŷ(x))
```
This requires solving a linear system `A * J = B`, where `A ∈ ℝᵈˣᵈ`, `B ∈ ℝᵈˣⁿ` and `J ∈ ℝᵈˣⁿ`.

# Fields:
- `forward::F`: callable of the form `x -> ŷ(x)`
- `conditions::C`: callable of the form `(x,y) -> F(x,y)`
- `linear_solver::L`: callable of the form `(A,b) -> u` such that `A * u = b`
"""
struct ImplicitFunction{F,C,L}
    forward::F
    conditions::C
    linear_solver::L
end

struct SolverFailureException <: Exception
    msg::String
end

ImplicitFunction(forward, conditions) = ImplicitFunction(forward, conditions, linsolve)

"""
    implicit(x)

Make [`ImplicitFunction{F,C,L}`](@ref) callable by applying `implicit.forward`.
"""
(implicit::ImplicitFunction)(x) = implicit.forward(x)

"""
    frule(rc, (_, dx), implicit, x)

Custom forward rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the Jacobian-vector product `Jv` by solving `Au = Bv` and setting `Jv = u`.
"""
function ChainRulesCore.frule(
    rc::RuleConfig, (_, dx), implicit::ImplicitFunction, x::AbstractArray{R}
) where {R<:Real}
    (; forward, conditions, linear_solver) = implicit

    y = forward(x)

    conditions_x(x̃) = conditions(x̃, y)
    conditions_y(ỹ) = -conditions(x, ỹ)

    A(dỹ) = frule_via_ad(rc, (NoTangent(), dỹ), conditions_y, y)[2]
    B(dx̃) = frule_via_ad(rc, (NoTangent(), dx̃), conditions_x, x)[2]

    b = B(unthunk(dx))
    dy, info = linear_solver(A, b)
    if iszero(info.converged)
        throw(SolverFailureException("Linear solver failed to converge"))
    end

    return y, dy
end

"""
    rrule(rc, implicit, x)

Custom reverse rule for [`ImplicitFunction{F,C,L}`](@ref).

We compute the vector-Jacobian product `Jᵀv` by solving `Aᵀu = v` and setting `Jᵀv = Bᵀu`.
"""
function ChainRulesCore.rrule(
    rc::RuleConfig, implicit::ImplicitFunction, x::AbstractArray{R}
) where {R<:Real}
    (; forward, conditions, linear_solver) = implicit

    y = forward(x)

    conditions_x(x̃) = conditions(x̃, y)
    conditions_y(ỹ) = -conditions(x, ỹ)

    Aᵀ = last ∘ rrule_via_ad(rc, conditions_y, y)[2]
    Bᵀ = last ∘ rrule_via_ad(rc, conditions_x, x)[2]

    function implicit_pullback(dy)
        u, info = linear_solver(Aᵀ, unthunk(dy))
        if iszero(info.converged)
            throw(SolverFailureException("Linear solver failed to converge"))
        end
        dx = Bᵀ(u)
        return (NoTangent(), dx)
    end

    return y, implicit_pullback
end
