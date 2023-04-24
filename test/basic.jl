# Basic use

using ForwardDiff
using ImplicitDifferentiation
using JET
using LinearAlgebra
using Random
using Test
using Zygote

Random.seed!(63);

function myidentity(x)
    a = [0.0]
    a[1] = first(x)
    return copy(x)
end;

x = rand(2, 1)

@testset verbose = true "Automatic diff fails" begin
    @test myidentity(x) ≈ x
    @test_throws MethodError ForwardDiff.jacobian(myidentity, rand(2))
    @test_throws ErrorException Zygote.jacobian(myidentity, rand(2))
end

forward(x) = myidentity(x), 0
conditions(x, y, z) = y .- x;
implicit = ImplicitFunction(forward, conditions);

@testset verbose = true "Implicit diff works" begin
    (first ∘ implicit)(x) ≈ x
    ForwardDiff.jacobian(first ∘ implicit, x) ≈ I
    Zygote.jacobian(first ∘ implicit, x)[1] ≈ I
end
