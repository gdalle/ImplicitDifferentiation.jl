## Imports

using Aqua
using Documenter
using ForwardDiff: ForwardDiff
using ImplicitDifferentiation
using JET
using JuliaFormatter
using Pkg
using Random
using Test
using Zygote: Zygote

DocMeta.setdocmeta!(
    ImplicitDifferentiation, :DocTestSetup, :(using ImplicitDifferentiation); recursive=true
)

function markdown_title(path)
    title = "?"
    open(path, "r") do file
        for line in eachline(file)
            if startswith(line, '#')
                title = strip(line, [' ', '#'])
                break
            end
        end
    end
    return title
end

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "examples")

## Test sets

@testset verbose = true "ImplicitDifferentiation.jl" begin
    @testset verbose = false "Code quality (Aqua.jl)" begin
        if VERSION >= v"1.9"
            Aqua.test_all(ImplicitDifferentiation; ambiguities=false)
        end
    end
    @testset verbose = true "Formatting (JuliaFormatter.jl)" begin
        @test format(ImplicitDifferentiation; verbose=false, overwrite=false)
    end
    @testset verbose = true "Static checking (JET.jl)" begin
        if VERSION >= v"1.9"
            JET.test_package(
                ImplicitDifferentiation;
                target_defined_modules=true,
                toplevel_logger=nothing,
            )
        end
    end
    @testset verbose = false "Doctests (Documenter.jl)" begin
        doctest(ImplicitDifferentiation)
    end
    @testset verbose = true "Examples" begin
        @info "Example tests"
        for file in readdir(EXAMPLES_DIR_JL)
            path = joinpath(EXAMPLES_DIR_JL, file)
            title = markdown_title(path)
            @info "$title"
            @testset verbose = true "$title" begin
                include(path)
            end
        end
    end
    @testset verbose = true "Errors" begin
        @info "Error tests"
        include("errors.jl")
    end
    @testset verbose = true "Systematic" begin
        @info "Systematic tests"
        include("systematic.jl")
    end
end
