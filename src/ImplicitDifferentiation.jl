module ImplicitDifferentiation

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig
using ChainRulesCore: frule_via_ad, rrule_via_ad, unthunk
using KrylovKit: linsolve

include("implicit_function.jl")

export ImplicitFunction

end
