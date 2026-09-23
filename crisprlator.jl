### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7104d3da-b768-11f1-911d-25e6d0f83c37
begin
	import Pkg; Pkg.activate(".")
	
	using GeneRegulatorySystems
	import GeneRegulatorySystems as GRS

	using DataFrames
	using CSV
	using JLD2
	using JSON
	using OrderedCollections: OrderedDict

	using Flux
	using NeuralEstimators
	
	using Statistics: mean, std
	import Distributions
	import Random
	import Bijectors
	
	using WGLMakie; WGLMakie.activate!()
	using PlutoUI
	import CytoscapeJS
	
	TableOfContents()
end

# ╔═╡ 90d0728c-27a2-4e10-9a32-9fc4e2f2c7fd
Base.show(io::IO, m::MIME"text/html", g::CytoscapeJS.Cytoscape) = show(io, m, CytoscapeJS.Bonito.App(g))

# ╔═╡ 49c565c1-1d38-4829-96b7-e8df709d9fd8
show_schedule(x; title = "schedule.json") = let
	x = JSON.parse(x)
	s = sprint(io -> JSON.print(io, x, 4))
	HTML("""
	<details open>
	  <summary style="cursor:pointer;font-family:monospace;font-weight:300;font-size:0.85em;padding:6px 0">$(title)</summary>
	  <div style="position:relative">
	    <button onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText);this.innerText='✓'"
	            style="position:absolute;top:6px;right:6px;font-size:0.75em;cursor:pointer">copy</button>
	    <pre style="max-height:400px;overflow:auto;font-size:0.8em;margin:0">$(replace(s, "<" => "&lt;"))</pre>
	  </div>
	</details>
	""")
end

# ╔═╡ 45c6866d-9c1c-473e-a50a-dfa07c122b23
md"""
## 1. Data
"""

# ╔═╡ c8e2bf3a-4b48-459c-b1db-a426b825b237
md"""Which trace: $(@bind trace_key Select([:closed => "Fig. 5 — intact ring (oscillates)", :open => "Supp. Fig. 6 — open-ring control (does not)"]))"""

# ╔═╡ 08c515a4-0ae5-4a7e-849f-70d204a84564
genes = [:n1, :n2, :n3]

# ╔═╡ 258ad098-5c2d-47da-8a92-8b0a0ca626f2
gene_colours = GRS.Visualisation.GroupColors(Dict(
    "n1" => colorant"crimson",
    "n2" => colorant"dodgerblue",
    "n3" => colorant"gold",
))


# ╔═╡ 1d832483-b6c6-46e5-937a-5bb8bae6446e
function load_trace(path, limits)
    trace = CSV.read(path, DataFrame)

    trace[!, 1] .*= 60.0
    rename!(trace, names(trace)[1] => :t)

    source_reporters = [
        "normalized mCherry fluorescence (%)",
        "normalized Cerulean fluorescence (%)",
        "normalized mCitrine fluorescence (%)",
    ]

    for (source, reporter, (lo, hi)) in zip(
        source_reporters,
        genes,
        limits,
    )
        percentage = Float64.(trace[!, source]) ./ 100
        trace[!, reporter] = lo .+ (hi - lo) .* percentage
    end

    select(trace, :t, genes...)
end

# ╔═╡ 372c40d6-fd81-4bc4-8e27-adb495123e1f
traces = let
    closed_limits = [
        (300.0, 700.0),
        (300.0, 700.0),
        (300.0, 700.0),
    ]

    open_limits = [
        (1800.0, 2200.0),
        (25.0, 135.0),
        (1400.0, 1800.0),
    ]

    Dict(
        :closed => load_trace(
            "data/crisprlator_fig5.csv",
            closed_limits,
        ),
        :open => load_trace(
            "data/crisprlator_suppfig6.csv",
            open_limits,
        ),
    )
end;

# ╔═╡ fa577a74-352b-4a65-9170-21b53c3ea45c
trace = traces[trace_key];

# ╔═╡ 0bb26999-08c8-4e2f-aec3-21aa2d432449
let
	f = Figure(size=(700, 350))
	ax = Axis(f[1,1], xlabel="time (s)", ylabel="relative fluorescence (a.u.)",
	          title="CRISPRlator, $(trace_key)", yzoomlock = true)
	for (col, lab, c) in zip(genes, ["mCherry (n1)","Cerulean (n2)","mCitrine (n3)"],
	                         [:crimson, :dodgerblue, :gold])
		lines!(ax, trace[!,1], trace[!,col]; color=c, label=lab)
	end
	axislegend(ax; position=:rt)
	f
end

# ╔═╡ 41a5b839-6f0c-4496-8e4c-2f06763bb337
md"""
## 2. Schedule
"""

# ╔═╡ 0cacc78b-3555-4767-a99a-14492e647dbe
md"""
### 2.1 Model definition
"""

# ╔═╡ 9157e633-75ad-4ea7-81d3-6c6618a26c1e
base_rates_spec = """
{
	"activation": 2.5,
	"deactivation": 10.0,
	"trigger": 6.6e-7,
	"abortion": 0.01,
	"transcription": 0.01,
	"mrna_decay": 0.001,
	"translation": 3.0e-9,
	"protein_decay": 1.0e-10
}
""";

# ╔═╡ dacfce12-0bcd-4db5-9be2-e860f634ff47
base_rates = JSON.parse(base_rates_spec; dicttype=Dict{Symbol, Float64})

# ╔═╡ 78c7a1ad-f8da-4f04-bbbb-49b594374aab
md"""Free topology: $(@bind free_topology Switch(default=true))"""

# ╔═╡ 99d1e474-40ff-421c-8dca-bd19448422a7
md"""
### 2.2 Experiment schedule definition
"""

# ╔═╡ 581ae951-195b-4fe6-883a-f4415a8ed5d0
md"""
Include open-ring experiment:
$(@bind include_open_circuit Switch(default=true))
"""

# ╔═╡ 2fbd463d-f27f-4547-8fbe-34d27baddf7d
conditions = include_open_circuit ? (:closed, :open) : (:closed,)

# ╔═╡ 2c5c5bb8-8004-4d04-9d0d-c71ffea676f2
md"""
### 2.3 Run simulation
"""

# ╔═╡ 6a746e77-2632-4ccc-bee1-893761dbe2a6
begin
	function run(schedule!::GRS.Schedule; seed="", kwargs...)
		schedule! = GRS.Scheduling.reseed(schedule!, seed)
		samples = NamedTuple[]
		lock = ReentrantLock()
		function trace(state = nothing; primitive! = nothing, _...)
			state === nothing && return
			bindings = primitive! === nothing ? Dict{Symbol, Any}() : primitive!.bindings
			counts = GRS.Models.counts(state)
			@lock lock push!(samples, (; 
								   t = GRS.Models.t(state),
								   condition = get(bindings, :condition, nothing),
								   rep = get(bindings, :rep, nothing),
								   x = [get(counts, Symbol(gene, ".proteins"), 0.0) for gene in genes]))
		end
		schedule!(GRS.Models.FlatState(); trace, kwargs...)
		samples
	end
	run(schedule!::GRS.Schedule, parameters::AbstractDict{Symbol};
	    path = "/.circuit", kwargs...) =
	    run(GRS.Models.remake(schedule!, path, parameters); kwargs...)
	
end

# ╔═╡ f5977006-3f4e-4204-9565-b4217d41f38a
function plot_samples(samples; genes, colours, unit=:s, selector = (_ -> true), show_reps=false, ylabel="proteins")
	units = Dict(:s => (1.0, "s"), :min => (60.0, "min"), :h => (3600.0, "h"))
    haskey(units, unit) || error("unit must be one of $(keys(units))")
    scale, ulabel = units[unit]
    selected = filter(selector, samples)
    isempty(selected) && error("no samples matched selector")

    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1]; xlabel = "time ($ulabel)", ylabel, yzoomlock = true)

    for (i, gene) in enumerate(genes)
        byt = Dict{Float64,Vector{Float64}}()
        for r in selected
            push!(get!(byt, r.t, Float64[]), r.x[i])
        end
        ts = sort!(collect(keys(byt)))
        t = ts ./ scale
        μ = [mean(byt[s]) for s in ts]
        colour = colours[gene]
    
        if show_reps
            byrep = Dict{Any,Vector{Tuple{Float64,Float64}}}()
            for r in selected
                push!(get!(byrep, r.rep, Tuple{Float64,Float64}[]), (r.t, r.x[i]))
            end
            for v in values(byrep)
                sort!(v; by = first)
                lines!(ax, [p[1] for p in v] ./ scale, [p[2] for p in v];
                       color = (colour, 0.15), linewidth = 0.8)
            end
        else
            σ = [length(byt[s]) > 1 ? std(byt[s]) : 0.0 for s in ts]
            band!(ax, t, μ .- σ, μ .+ σ; color = (colour, 0.10))
        end
    
        lines!(ax, t, μ; color = colour, label = String(gene))
    end


    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 085f902b-48d8-4c76-a7a0-56a95b8da0ee
@bind cond_key Select(["closed", "open"])

# ╔═╡ c2ba1b94-34d8-4e16-a799-4a4177bea7ab
md"""
## 3. Prior
"""

# ╔═╡ d9349b15-5d62-478e-92de-d672cbf86f18
begin
slots(definition; kinds = (:activation, :repression)) = [
    (; target = gene.name, kind, from = slot.from, at = slot.at, k = slot.k, w = slot.w)
    for gene in definition.genes
    for kind in kinds
    for slot in getfield(gene, kind).slots
]

slots(model::GRS.Models.Model; kwargs...) = slots(model.definition; kwargs...)

end

# ╔═╡ a1b50041-e44c-4acb-96fb-e433ab21a560
md"""
## A.1. GRS.jl extensions
"""

# ╔═╡ 1d06f01f-9fa2-4a7f-bb22-fe846cd7bff3
begin
    function disable_reaction(specification::AbstractDict{Symbol})
        disable_reaction(
            specification[:of];
            reaction=Symbol(specification[:reaction]),
        )
    end

    function disable_reaction(model; reaction)
        parameter = Symbol("reaction.$reaction.k⁺")
        GRS.Models.remake(model, Dict(parameter => 0.0))
    end

    
end

# ╔═╡ 93e6f519-3599-46b8-af04-d4a6eb0dd767
GRS.Specifications.constructor(
        ::Val{Symbol("regulation/v1/disable-reaction")},
    ) = disable_reaction

# ╔═╡ 94ac974b-346d-4722-859c-281dbbbf1064
function free_crispri_topology_spec(
    circuit_spec::AbstractString;
    kinds,
    allow_self=false,
    at=10.0,
    k=-1.0,
    p=-4.0,
)
    k_magnitude = -k
    specification = JSON.parse(
        circuit_spec;
        dicttype=OrderedDict{Symbol,Any},
    )

    circuit = specification[Symbol("{regulation/v1}")]
    genes = circuit[:genes]
    gene_names = String.(getindex.(genes, Ref(:name)))

    regulators = Dict(
        gene => "$gene.sgrna_cas9"
        for gene in gene_names
    )

    for target_gene in genes
        target = String(target_gene[:name])

        for kind in kinds
            existing = get(target_gene, kind, nothing)

            existing_slots =
                existing isa AbstractDict ? get(existing, :slots, []) :
                existing isa AbstractVector ? existing :
                Any[]

            slots_by_source = Dict(
                String(slot[:from]) => slot
                for slot in existing_slots
            )

            candidate_sources = [
                regulators[source]
                for source in gene_names
                if allow_self || source != target
            ]

            slots = map(candidate_sources) do source
                if haskey(slots_by_source, source)
                    slot = deepcopy(slots_by_source[source])
                    slot[:w] = get(slot, :w, 1.0)
                    slot
                else
                    OrderedDict{Symbol,Any}(
                        :from => source,
                        :at   => at,
                        :k    => -abs(k_magnitude),
                        :w    => 0.0,
                    )
                end
            end

            target_gene[kind] = OrderedDict{Symbol,Any}(
                :slots     => slots,
                :aggregate => "generalized_mean",
                :p         => p,
            )
        end
    end

    JSON.json(specification)
end

# ╔═╡ e11643e7-dfa8-4d4f-9a3f-55fa7aa75d80
circuit_spec = let
	at = 10.0
	k = -1.4
	kmake = 0.001
	kdecay = 2.0e-4
	spec = """
{
"{regulation/v1}": {
	"species": ["mrnas", "proteins"],
	"method": "HybridTau",
	"genes": [
		{"name": "n1", "color": "#e03c4e",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n3.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": -4.0}},
		{"name": "n2", "color": "#3c8ce0",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n1.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": -4.0}},
		{"name": "n3", "color": "#e0b93c",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n2.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": -4.0}}
		],
	"reactions": [
		{"name": "make_n1.sgrna_cas9", "from": ["n1.mrnas"], "to": ["n1.mrnas", "n1.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n1sgrna_cas9", "from": ["n1.sgrna_cas9"], "to": [], "rate": $kdecay},
		{"name": "make_n2.sgrna_cas9", "from": ["n2.mrnas"], "to": ["n2.mrnas", "n2.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n2sgrna_cas9", "from": ["n2.sgrna_cas9"], "to": [], "rate": $kdecay},
		{"name": "make_n3.sgrna_cas9", "from": ["n3.mrnas"], "to": ["n3.mrnas", "n3.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n3.sgrna_cas9", "from": ["n3.sgrna_cas9"], "to": [], "rate": $kdecay}
	]
}
}
"""
	free_topology ? free_crispri_topology_spec(spec; kinds=(:repression,:activation), at=1000.0, k, p=-4.0, allow_self=true) : spec
end;

# ╔═╡ 3bcac3d3-e431-41ea-a525-74082d329fcd
circuit = GRS.Models.parse(circuit_spec; base_rates=base_rates);

# ╔═╡ 0dfa8576-3e9f-44ac-a13d-2539dfb4f898
CytoscapeJS.Cytoscape(GRS.Visualisation.Network(circuit); group_colors = gene_colours, height="500px")

# ╔═╡ b20ad086-2b04-4cc2-8d08-4dbc3fad222b
prior_setup = let
    edges = slots(circuit)
    pairs = unique((e.from, e.target) for e in edges)
    pair_index = Dict(p => i for (i, p) in enumerate(pairs))
    free = Dict(p => length(unique(e.kind for e in edges if (e.from, e.target) == p)) > 1 for p in pairs)

    names = reduce(vcat, [[Symbol("edge_$i.w"), Symbol("edge_$i.at")] for i in eachindex(pairs)])
    prior = Distributions.product_distribution(reduce(vcat,
        [[Distributions.Uniform(-1, 1), Distributions.LogNormal(log(30.0), 1.0)]
         for _ in eachindex(pairs)]))

    ksharp, guide_decay = 1.4, 2.0e-4

    function decode(θ)
        values = Dict{Symbol,Float64}()
        for e in edges
            i = pair_index[(e.from, e.target)]
            w, at = θ[2i - 1], θ[2i]
            weight = free[(e.from, e.target)] ?
                (e.kind == :activation ? max(w, 0.0) : max(-w, 0.0)) : abs(w)
            stem = "$(e.target).$(e.kind).$(e.from)"
            values[Symbol("$stem.at")] = at
            values[Symbol("$stem.k")]  = -ksharp
            values[Symbol("$stem.w")]  = weight
        end
        for gene in genes
            values[Symbol("reaction.decay_$(gene)sgrna_cas9.k⁺")] = guide_decay
        end
        values
    end

    (; names, prior, decode, edges, pairs)
end


# ╔═╡ c508c880-7466-4eaa-9dbe-c0c1aee95da2
θ = rand(prior_setup.prior)

# ╔═╡ 79cfc878-44a5-4c32-a5a1-83d11b2c261e
schedule_spec = let
    num_replicates = 10
    sample_rate = 600
    burnin_closed = first(traces[:closed].t) - sample_rate
    burnin_open   = first(traces[:open].t) - sample_rate
    culture_closed = last(traces[:closed].t) - first(traces[:closed].t) + sample_rate
    culture_open = last(traces[:open].t) - first(traces[:open].t) + sample_rate
    

    open_spec = :open in conditions ?
    """
    ,
    {
        "condition": "open",
        "do": {
            "{regulation/v1/disable-reaction}": {
                "of": {"\$": "circuit"},
                "reaction": "make_n3.sgrna_cas9"
            }
        },
        "step": [
            {"stage": "burn_in", "to": $burnin_open, "step": $sample_rate},
            {
                "stage": "culture",
                "each": {"length": $num_replicates},
                "as": "rep",
                "branch": true,
                "step": {"to": $culture_open, "step": $sample_rate}
            }
        ]
    }
    """ : ""

    """
    {
        "seed": "\${rootseed}",
        "base_rates": $base_rates_spec,
        "step": {
            "circuit": $circuit_spec,
            "step": [
                {"{add}": {"\$": ["defaults", "bootstrap"]}},
                {
                    "branch": true,
                    "step": [
                        {
                            "condition": "closed",
                            "do": {"\$": "circuit"},
                            "step": [
                                {"stage": "burn_in", "to": $burnin_closed, "step": $sample_rate},
                                {
                                    "stage": "culture",
                                    "each": {"length": $num_replicates},
                                    "as": "rep",
                                    "branch": true,
                                    "step": {"to": $culture_closed, "step": $sample_rate}
                                }
                            ]
                        }
                        $open_spec
                    ]
                }
            ]
        }
    }
    """
end;

# ╔═╡ 40c3083b-4ab2-4017-bb30-bcd8662d9846
show_schedule(schedule_spec; title="crisprlator.schedule.json")

# ╔═╡ 140134ca-0d29-40e0-9774-6a25be61edfa
schedule! = let
	s = GRS.Models.parse(schedule_spec)
	GRS.Scheduling.prebuild(s, "/.circuit")
end

# ╔═╡ 8cc0bb7a-5dfa-47ea-8037-193b5c9fe5ef
samples = run(schedule!)

# ╔═╡ 7ed69734-84a1-4e94-9a46-7d98c970d087
plot_samples(samples; genes, colours=gene_colours, unit = :h, selector = r -> r.condition == cond_key, show_reps=false)

# ╔═╡ a6d2d008-93cb-4a8e-b041-f9561f39c39b
rows = run(schedule!, prior_setup.decode(θ); seed = "1")

# ╔═╡ Cell order:
# ╟─90d0728c-27a2-4e10-9a32-9fc4e2f2c7fd
# ╟─49c565c1-1d38-4829-96b7-e8df709d9fd8
# ╠═7104d3da-b768-11f1-911d-25e6d0f83c37
# ╟─45c6866d-9c1c-473e-a50a-dfa07c122b23
# ╟─c8e2bf3a-4b48-459c-b1db-a426b825b237
# ╠═08c515a4-0ae5-4a7e-849f-70d204a84564
# ╟─258ad098-5c2d-47da-8a92-8b0a0ca626f2
# ╟─1d832483-b6c6-46e5-937a-5bb8bae6446e
# ╟─372c40d6-fd81-4bc4-8e27-adb495123e1f
# ╟─fa577a74-352b-4a65-9170-21b53c3ea45c
# ╟─0bb26999-08c8-4e2f-aec3-21aa2d432449
# ╟─41a5b839-6f0c-4496-8e4c-2f06763bb337
# ╟─0cacc78b-3555-4767-a99a-14492e647dbe
# ╠═9157e633-75ad-4ea7-81d3-6c6618a26c1e
# ╠═dacfce12-0bcd-4db5-9be2-e860f634ff47
# ╟─78c7a1ad-f8da-4f04-bbbb-49b594374aab
# ╠═e11643e7-dfa8-4d4f-9a3f-55fa7aa75d80
# ╠═3bcac3d3-e431-41ea-a525-74082d329fcd
# ╠═0dfa8576-3e9f-44ac-a13d-2539dfb4f898
# ╟─99d1e474-40ff-421c-8dca-bd19448422a7
# ╟─581ae951-195b-4fe6-883a-f4415a8ed5d0
# ╟─2fbd463d-f27f-4547-8fbe-34d27baddf7d
# ╠═79cfc878-44a5-4c32-a5a1-83d11b2c261e
# ╠═40c3083b-4ab2-4017-bb30-bcd8662d9846
# ╠═140134ca-0d29-40e0-9774-6a25be61edfa
# ╟─2c5c5bb8-8004-4d04-9d0d-c71ffea676f2
# ╠═6a746e77-2632-4ccc-bee1-893761dbe2a6
# ╠═8cc0bb7a-5dfa-47ea-8037-193b5c9fe5ef
# ╠═f5977006-3f4e-4204-9565-b4217d41f38a
# ╟─085f902b-48d8-4c76-a7a0-56a95b8da0ee
# ╠═7ed69734-84a1-4e94-9a46-7d98c970d087
# ╟─c2ba1b94-34d8-4e16-a799-4a4177bea7ab
# ╟─d9349b15-5d62-478e-92de-d672cbf86f18
# ╠═b20ad086-2b04-4cc2-8d08-4dbc3fad222b
# ╠═c508c880-7466-4eaa-9dbe-c0c1aee95da2
# ╠═a6d2d008-93cb-4a8e-b041-f9561f39c39b
# ╟─a1b50041-e44c-4acb-96fb-e433ab21a560
# ╟─1d06f01f-9fa2-4a7f-bb22-fe846cd7bff3
# ╠═93e6f519-3599-46b8-af04-d4a6eb0dd767
# ╟─94ac974b-346d-4722-859c-281dbbbf1064
