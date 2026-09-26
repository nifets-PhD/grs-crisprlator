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
	using Flux.Optimisers: OptimiserChain, ClipNorm, Adam, WeightDecay
	using NeuralEstimators
	
	using Statistics: mean, std, cor, median, quantile
	import Distributions
	import Random
	import Bijectors
	using LinearAlgebra
	
	using CairoMakie, WGLMakie
	CairoMakie.activate!()
	using PlutoUI
	import CytoscapeJS
	CytoscapeJS.Bonito.Page()

	
	TableOfContents()
end

# ╔═╡ f030a2b8-7aa1-49d0-a438-d46d06870998
md"""
# Posterior estimation of the CRISPRlator synthetic genetic circuit

Using GeneRegulatorySystems.jl and NeuralEstimators.jl
"""

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

# ╔═╡ 14914584-7bbd-47d0-a9ba-bde30464ceef
md"""
Caveat: the authors of the study normalised the readout per gene instead of globally, which removes much of the signal. To make inference tractable at all here, I was force to "guess" the relative differences between the three genes. This information would be avaialble from the experiment itself, it's just about undoing one of the postprocessing steps.
"""

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
### 2.1. Model definition
"""

# ╔═╡ 9157e633-75ad-4ea7-81d3-6c6618a26c1e
base_rates_spec = """
{
	"activation": 2.5,
	"deactivation": 10.0,
	"trigger": 6.6e-6,
	"abortion": 0.01,
	"transcription": 0.04,
	"mrna_decay": 0.003,
	"translation": 5.0e-8,
	"protein_decay": 2.9e-10
}
""";

# ╔═╡ dacfce12-0bcd-4db5-9be2-e860f634ff47
base_rates = JSON.parse(base_rates_spec; dicttype=Dict{Symbol, Float64})

# ╔═╡ 78c7a1ad-f8da-4f04-bbbb-49b594374aab
md"""Free topology: $(@bind free_topology Switch(default=true))"""

# ╔═╡ 99d1e474-40ff-421c-8dca-bd19448422a7
md"""
### 2.2. Experiment schedule definition
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
### 2.3. Run simulation
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

# ╔═╡ 3394482e-6652-4736-81ea-5a5b6daa82f8
md"""show replicates: $(@bind show_reps Switch(default=true))"""


# ╔═╡ f5977006-3f4e-4204-9565-b4217d41f38a
function plot_samples(samples, condition; unit=:h, show_reps=false, ylabel="proteins")
    units = Dict(:s => (1.0, "s"), :min => (60.0, "min"), :h => (3600.0, "h"))
	haskey(units, unit) || error("unit must be one of $(keys(units))")
    scale, ulabel = units[unit]
    selected = filter(r -> r.condition == String(condition), samples)
    isempty(selected) && error("no samples for $condition")

    fig = Figure(size = (680, 300))
    ax = Axis(fig[1, 1]; xlabel = "time ($ulabel)", ylabel, title = String(condition), yzoomlock = true)

    for (i, gene) in enumerate(genes)
        byt = Dict{Float64,Vector{Float64}}()
        for r in selected
            push!(get!(byt, r.t, Float64[]), r.x[i])
        end
        ts = sort!(collect(keys(byt)))
        t = ts ./ scale
        μ = [mean(byt[s]) for s in ts]
        colour = gene_colours[gene]
    
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

# ╔═╡ 57e7638a-3660-4c13-9e83-f833e9e98ed7
md"""
### 2.4. Compare output against data

Since our data is population level, we have to average out the simulated branches, each of which represents one bacteria. We also discard the burn-in samples. 
"""

# ╔═╡ 491c418e-c7bc-431b-8ce3-0512932b1784
begin
	normalise(v) = let (lo, hi) = extrema(v)
		hi == lo ? zero(v) : (v .- lo) ./ (hi - lo)
	end

	normalise(nt::NamedTuple) = let (lo, hi) = extrema(Iterators.flatten(nt))
		map(m -> hi == lo ? zero(m) : (m .- lo) ./ (hi - lo), nt)
	end
end

# ╔═╡ 489adcea-561c-45ae-8a35-b7aa4435cd6e
function observation(samples)
	normalise(NamedTuple{conditions}(map(conditions) do c
		times = traces[c].t
		index = Dict(round(Int, t) => i for (i, t) in enumerate(times))
		values = zeros(length(times), length(genes))
		n = zeros(Int, length(times))
		for s in samples
			s.condition == string(c) || continue
			i = get(index, round(Int, s.t), 0)
			i == 0 && continue
			n[i] += 1
			values[i, :] .+= s.x
		end
		any(iszero, n) && error("no samples at t = $(times[n .== 0]) for $c")
		values ./ n
	end))
end

# ╔═╡ 4aa90a45-6339-42aa-9d59-90394f8fd4e3
x = normalise(NamedTuple{conditions}(map(c -> Matrix{Float64}(traces[c][:, collect(genes)]), conditions)))

# ╔═╡ cf3a114a-c147-40cf-9709-0ab901e74cae
function plot_observation(condition, simulated, observed)
	t = traces[condition].t
	fig = Figure(size = (680, 300))
	ax = Axis(fig[1,1]; xlabel="time (s)", ylabel="relative fluorescence", title=String(condition), yzoomlock=true)
	for (j, gene) in enumerate(genes)
		colour = gene_colours[gene]
		lines!(ax, t, observed[:, j]; color=colour, linewidth=2, linestyle=:dash, label="$gene data")
		lines!(ax, t, simulated[:, j]; color=colour, linewidth=2, label="$gene sim")
	end
	axislegend(ax; position=:rt)
	fig
end

# ╔═╡ c2ba1b94-34d8-4e16-a799-4a4177bea7ab
md"""
## 3. Prior
"""

# ╔═╡ 3375089b-595a-4258-b50c-c6812b604587
md"""
### 3.1. Define prior
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

# ╔═╡ ede66989-a2c4-43c2-8640-5944075d43e6
md"""
We parametrise the network via sᵢⱼ ∈ \[-1,1\], normalised regulation strength from gene i to gene j, positive if activation and negative if repression. 
"""

# ╔═╡ d8198379-3f20-48ae-8ccd-a03ab4bcd714
edge_label(n) = let p = Base.split(String(n), ".edge.")
	"$(p[1]) ← $(replace(p[2], ".sgrna_cas9" => ""))"
end

# ╔═╡ 8c1554a1-1638-47f1-adfb-23c383923d95
@bind resample PlutoUI.Button("resample θ")

# ╔═╡ 452a3724-e706-4881-8f7a-4c0083e80c76
md"""
### 3.2. Generate (θ, Z) training samples from prior distribution
"""

# ╔═╡ 58274dc3-1edb-47af-b2fb-09585df8b474
flatten(x) = Float32.(reduce(vcat, x))

# ╔═╡ f2a19c0f-c339-44bf-9773-98df6ccce7c1
md"""
## 4. Posterior estimation
"""

# ╔═╡ 8a3d393c-1bae-4179-91c5-fdc2083e43f9
md"""
### 4.1. Train a Neural Posterior Estimator to learn p(θ | x)
"""

# ╔═╡ df2878c1-1b00-4c1a-896d-a205dcd4cdad
md"""
Using a ConvNet allow us to extract features from both the time and the frequency domain, which is especially important for this oscillation example. 
"""

# ╔═╡ 6d5f497d-da8b-49ea-b3d3-ef0c8d192a03
function embedding(; widths = (8,16,16), strides=(2,2,4,4), pool=3, out=16, k=7, channels=3)
	sizes = (channels, widths...)
	convs  = [Conv((k,), i => o, gelu; stride=s, pad = :same) for (i,o,s) in zip(sizes, widths, strides)]
	Chain(convs..., AdaptiveMeanPool((pool,)), Flux.flatten, Dense(last(widths) * pool => out))
end

# ╔═╡ d641081b-f7ce-475e-b0e4-c1422403b51f
md"""
We can now train the network:
"""

# ╔═╡ 7d515807-3dc0-43af-9341-e971d6f270a9
function with_live_logs(f)
	old = stdout
	rd, wr = redirect_stdout()
	t = @async for line in eachline(rd)
		@info line
	end
	try
		f()
	finally
		redirect_stdout(old)
		close(wr)
		close(rd)
	end
end

# ╔═╡ 4e692e0d-d6ba-44a4-aeed-135914a56a84
md"""
### 4.2. Simulation Based Calibration

Let us first check whether the learned posterior P is well calibrated (i.e. whether it behaves like the true posterior would). The idea is to take our validation samples (θ, Z) that the estimator has not seen, draw several θ̂ ~ P(⋅|Z) samples from the learned posterior, and see where for each coordinate of θ, where it ranks among θ̂. If we do this for several samples, we expect the rank per coordinate to be uniformly distributed.
"""

# ╔═╡ aef9cd8f-5a72-445f-9cfd-14ab2959df8f
function sbc_plot(result, simulations; alpha = 0.05)
	d, m = size(result.ranks)
	c = sqrt(-log(alpha / 2) / 2) / sqrt(m)
	u = range(0, 1; length = m)
	fig = Figure(size = (680, 230 * cld(d, 3)))
	for k in 1:d
		ax = Axis(fig[fld1(k, 3), mod1(k, 3)];
		          title = edge_label(simulations.names[k]), xlabel = "rank", ylabel = "ECDF − uniform")

		band!(ax, [0, 1], [-c, -c], [c, c]; color = (:grey, 0.25))
		hlines!(ax, [0.0]; color = :red, linestyle = :dash)
		lines!(ax, u, sort(result.ranks[k, :]) .- u; color = :steelblue)
	end
	fig
end

# ╔═╡ 237184cb-50cd-40ed-ad78-9c8bbc969804
md"""
Also check how much the posterior distribution shrinks compared to the prior:
"""

# ╔═╡ d7610be6-5d7c-427a-95b1-82d903ac4282
md"""
### 4.3. Condition the learned posterior on the experimental data
"""

# ╔═╡ 9621d07a-7fb9-4e32-863e-b57a9e9b8a11
md"""
#### 4.3.1. Posterior predictive checks
"""

# ╔═╡ 4abd251a-1de6-4750-be13-6067cc7e2e92
function plot_ppc_lines(condition, draws, observed; n = 12)
	t = traces[condition].t
	fig = Figure(size = (680, 440))
	for (j, gene) in enumerate(genes)
		colour = gene_colours[gene]
		ax = Axis(fig[j,1]; ylabel = "relative fluorescence",
				  xlabel = j == length(genes) ? "time (s)" : "",
				  title = j == 1 ? String(condition) : "", yzoomlock = true)
		for d in draws[1:min(n, end)]
			lines!(ax, t, d[condition][:, j]; color = (colour, 0.3), linewidth = 1,
				   label = "$gene sim")
		end
		lines!(ax, t, observed[:, j]; color = colour, linewidth = 2, linestyle = :dash,
			   label = "$gene data")
		axislegend(ax; position = :rt, merge = true)
		j < length(genes) && hidexdecorations!(ax; grid = false)
	end
	rowgap!(fig.layout, 4)
	fig
end


# ╔═╡ 409aa215-cc29-4469-bf6e-f6a59e8fb844
md"""
### 4.4. Have we recovered the CRISPRLATOR network topology?
"""

# ╔═╡ 08c0dbaf-a345-44a1-adc6-16e7f08fbb26
md"""
#### 4.4.1. Marginal densities
"""

# ╔═╡ 33a08a96-7161-430e-b89a-29fb6fa6087e
md"""
Blue: ring edges, Red: reverse ring edges. Sign of s tells us if the edge is activating or repressing, and magnitude of s gives the strength of the interaction.
"""

# ╔═╡ 03c97a0a-e488-4026-b4a5-c6fc6aa55bb4
md"""
This density plots for the ring edges shows that the n3 -| n1 edge is most constrained by the data (as the interventional data we have is exactly for this edge).
"""

# ╔═╡ a56d437c-7605-4e7e-b78c-19e17f8102f8
md"""
#### 4.4.2. Bayes Factor
"""

# ╔═╡ 734acf9f-030f-49bd-8c47-55580e4421e9
bayes_factor(p, p0) = (p / (1 - p)) / (p0 / (1 - p0))

# ╔═╡ 91fff83d-ff9d-4dbf-a3e8-04d14771ef1b
md"""
### 4.5. Ablation study: what happens if we don't use the interventional data?
"""

# ╔═╡ 96560e70-0ddc-4e1c-ad6c-f63602e99993
x_wt = Float32.(normalise(Matrix{Float64}(traces[:closed][:, collect(genes)])));

# ╔═╡ fd2da864-6c6b-4674-95f4-f56b4424d2d7
md"""
### 4.6. Synthetic ring sanity check

If we actually know the ground truth regulation, can we recover it from the observed dynamics?
"""

# ╔═╡ 1653db04-b38a-4f1a-857b-3b79f2e08584
md"""
We see here the same trend where the ring edges are well recovered but the reverse ring positive edges , showing that the data we have is not enough to fully constrain the topology. Of course, in this case we know from the construciton that all edges ought to be repressive, but in a more general setting we wouldn't.
"""

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
	at = 700.0
	k = -2.3
	kmake = 0.003
	kdecay = 1e-4
	
	p = -4.0
	spec = """
{
"{regulation/v1}": {
	"species": ["mrnas", "proteins"],
	"method": "HybridTau",
	"genes": [
		{"name": "n1", "color": "#e03c4e",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n3.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": $p}},
		{"name": "n2", "color": "#3c8ce0",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n1.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": $p}},
		{"name": "n3", "color": "#e0b93c",
			"base_rates": {"\$": "base_rates"},
			"repression": {"slots": [{"from": "n2.sgrna_cas9", "at": $at, "k": $k, "w": 1.0}], "aggregate": "generalized_mean", "p": $p}}
		],
	"reactions": [
		{"name": "make_n1.sgrna_cas9", "from": ["n1.mrnas"], "to": ["n1.mrnas", "n1.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n1.sgrna_cas9", "from": ["n1.sgrna_cas9"], "to": [], "rate": $kdecay},
		{"name": "make_n2.sgrna_cas9", "from": ["n2.mrnas"], "to": ["n2.mrnas", "n2.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n2.sgrna_cas9", "from": ["n2.sgrna_cas9"], "to": [], "rate": $kdecay},
		{"name": "make_n3.sgrna_cas9", "from": ["n3.mrnas"], "to": ["n3.mrnas", "n3.sgrna_cas9"], "rate": $kmake},
		{"name": "decay_n3.sgrna_cas9", "from": ["n3.sgrna_cas9"], "to": [], "rate": $kdecay}
	]
}
}
"""
	free_topology ? free_crispri_topology_spec(spec; kinds=(:repression,:activation), at=200000.0, k, p, allow_self=true) : spec
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

    names = [Symbol("$(target).edge.$(from)") for (from, target) in pairs]
    prior = Distributions.product_distribution([Distributions.Uniform(-1, 1) for _ in eachindex(pairs)])

    at_low, at_high = 50.0, 200000.0
    s_off = 0.1

    function decode(θ)
        values = Dict{Symbol,Float64}()
        for e in edges
            s = θ[pair_index[(e.from, e.target)]]
            stem = "$(e.target).$(e.kind).$(e.from)"
            values[Symbol("$stem.at")] = at_high * (at_low / at_high)^abs(s)
            on = abs(s) >= s_off && (s > 0) == (e.kind == :activation)
            values[Symbol("$stem.w")]  = on ? 1.0 : 0.0
        end
        values
    end

    (; names, prior, decode, edges, pairs, s_off, at_low, at_high)
end


# ╔═╡ 620088a8-1700-4938-9bc7-a647a3beabf9
prior_setup.names

# ╔═╡ c508c880-7466-4eaa-9dbe-c0c1aee95da2
θ = begin
	resample
	rand(prior_setup.prior)
end

# ╔═╡ e540af28-9eb4-439a-9e98-800552a89aed
function standardise(θ, ref)
	b = Bijectors.bijector(prior_setup.prior)
	push(x) = reduce(hcat, b.(eachcol(Float64.(x))))
	u = push(ref)
	Float32.((push(θ) .- mean(u; dims=2)) ./ std(u; dims=2))
end

# ╔═╡ 3e493391-0720-4ecc-94c6-a93a03cfdb1f
function train_posterior(θ, Z, split; seed="1", out=16, layers=4, width=64, lr=1e-3, batchsize=128, epochs=500, patience=50, sigma=0.08f0, path=joinpath("results", "estimator_$(seed).jld2"))
	mkpath(dirname(path))
	estimator = PosteriorEstimator(embedding(; out), NormalisingFlow(size(θ, 1), out; num_coupling_layers=layers, width))
	isfile(path) && return Flux.loadmodel!(estimator, load(path, "state"))

	Random.seed!(hash(seed))
	θs = standardise(θ, θ[:, split.train])
	θtrain, θval = θs[:, split.train], θs[:, split.val]
	Ztrain, Zval = Z[:, :, split.train], Z[:, :, split.val]
	sim = let d = IdDict(θtrain => Ztrain, θval => Zval)
		θ -> let Zf = d[θ]; Zf .+ sigma .* randn(Float32, size(Zf)) end
	end
	estimator = train(estimator, θtrain, θval, sim;
		 optimiser=OptimiserChain(ClipNorm(5.0), Adam(lr), WeightDecay(1e-4)),
		 batchsize, epochs, stopping_epochs = patience, epochs_per_Z_refresh = 1,
		 use_gpu=false, savepath=nothing, verbose=true)
	jldsave(path; state=Flux.state(estimator))
	estimator
end

# ╔═╡ 8b29431d-bd0d-468c-8e99-4cf85c2cd945
function sbc(estimator, θ, Z, split; nsims=1500, L=400)
	Zval = Z[:, :, split.val]
	θval = standardise(θ[:, split.val], θ[:, split.train])
	d = size(θ, 1)
	m = min(nsims, length(split.val))
	ranks = reduce(hcat, map(1:m) do j
		P = sampleposterior(estimator, reshape(Zval[:, :, j], size(Zval)[1:2]..., 1); N=L)
		draws = P isa AbstractVector ? P[1] : P
		[mean(@view(draws[k, :]) .< θval[k, j]) for k in 1:d]
	end)
	uniform = range(0, 1; length = m)
	(; cov90 = [mean(0.05 .<= ranks[k, :] .<= 0.95) for k in 1:d],
	cov50 = [mean(0.25 .<= ranks[k, :] .<= 0.75) for k in 1:d], 
	 	maxrankdev = [maximum(abs.(sort(ranks[k, :]) .- uniform)) for k in 1:d],
	 	ranks)
end

# ╔═╡ d3265f2f-a2d4-452a-a284-4510808440c2
function unstandardise(z, ref)
	b = Bijectors.bijector(prior_setup.prior)
	u = reduce(hcat, b.(eachcol(Float64.(ref))))
	scaled = Float64.(z) .* std(u; dims=2) .+ mean(u; dims=2)
	reduce(hcat, Bijectors.inverse(b).(eachcol(scaled)))
end

# ╔═╡ 461d24fe-3903-4889-89df-1d451b94d1db
function posterior(estimator, obs, ref; N=20_000)
	Z = reshape(obs, size(obs)..., 1)
	P = sampleposterior(estimator, Z; N)
	unstandardise(Float32.(P isa AbstractVector ? P[1] : P), ref)
end

# ╔═╡ c0cb2a78-a875-4416-b7b6-3dd70eaaab45
function shrinkage(estimator, θ, Z, split; nsims = 300, L = 400)
	idx = split.val[1:min(nsims, length(split.val))]
	ref = θ[:, split.train]
	sds = reduce(hcat, map(idx) do j
		vec(std(posterior(estimator, Z[:, :, j], ref; N = L); dims = 2))
	end)
	vec(mean(sds; dims = 2)) ./ vec(std(ref; dims = 2))
end

# ╔═╡ e91572ab-6858-4202-854e-10db4fa0e046
CytoscapeJS.Cytoscape(
	GRS.Visualisation.Network(GRS.Models.remake(circuit, prior_setup.decode(θ)));
	group_colors = gene_colours, height = "400px"
)

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
samples = run(schedule!; seed="2")

# ╔═╡ 7ed69734-84a1-4e94-9a46-7d98c970d087
PlutoUI.ExperimentalLayout.vbox([plot_samples(samples, c; show_reps) for c in conditions])

# ╔═╡ c3a4b23f-d7b3-471c-98cc-760cd90919d2
x̂ = observation(samples)

# ╔═╡ f1b39cf9-8f3e-4e62-a64b-7f7d8dddb661
PlutoUI.ExperimentalLayout.vbox([plot_observation(c, x̂[c], x[c]) for c in conditions])

# ╔═╡ a6d2d008-93cb-4a8e-b041-f9561f39c39b
samples_prior = run(schedule!, prior_setup.decode(θ); seed = "1")

# ╔═╡ c19bd5f5-2462-4840-9a92-e096c68dd5a6
PlutoUI.ExperimentalLayout.vbox([plot_observation(c, observation(samples_prior)[c], x[c]) for c in conditions])

# ╔═╡ 5edca0b9-0924-4da5-a159-12911761a06b
function sample_posterior(S; seed = "1", i = rand(1:size(S, 2)))
	θ = S[:, i]
	smp = run(schedule!, prior_setup.decode(θ); seed)
	obs = observation(smp)
	PlutoUI.ExperimentalLayout.vbox(vcat(
		[plot_observation(c, obs[c], x[c]) for c in conditions],
		CytoscapeJS.Cytoscape(
			GRS.Visualisation.Network(GRS.Models.remake(circuit, prior_setup.decode(θ)));
			group_colors = gene_colours, height = "400px")))
end

# ╔═╡ 3872327a-c1ce-4923-8dbd-758d31f73e5e
function generate(n; seed="1", path=joinpath("results", "simulations_$(n)_$(seed).jld2"), save_every=2500)
	mkpath(dirname(path))
	isfile(path) && return path
	
	rng = Random.Xoshiro(hash(seed))
	θ = rand(rng, prior_setup.prior, n)
	Z = NamedTuple{conditions}(map(c -> Array{Float32, 3}(undef, length(traces[c].t), length(genes), n), conditions))

	save(i) = jldsave(path; theta=θ[:, 1:i], n=i, seed, Z=map(z -> z[:, :, 1:i], Z), names=prior_setup.names, spec=circuit_spec, times=NamedTuple{conditions}(map(c -> traces[c].t, conditions)), genes)

	t0 = time()
	for i in 1:n
		o = observation(run(schedule!, prior_setup.decode(θ[:, i]); seed="$(seed)_$i"))
		for c in conditions
			Z[c][:, :, i] = o[c]
			i % save_every == 0 && (@info "sims" i elapsed = round(time() - t0; digits=1); save(i))
		end
	end
	save(n)
	path
end

# ╔═╡ 6e977053-1104-48eb-ac30-ae59237782c0
sim_path = generate(50000; seed="apple50k")

# ╔═╡ 66af873c-7d69-45e0-9f62-bb4b5669804c
simulations = let d = load(sim_path)
	(; θ = d["theta"],
	   Z = Float32.(reduce((a, b) -> cat(a, b; dims = 1), d["Z"][c] for c in conditions)),
	   names = d["names"], n = d["n"], seed = d["seed"], times = d["times"])
end


# ╔═╡ 6b5dc235-83af-4584-93b4-fbea77b50674
split = let m = simulations.n, ntrain = round(Int, 0.9m)
	(; train=1:ntrain, val=(ntrain+1):m)
end

# ╔═╡ 0ceab4f6-7249-4dc9-a1aa-604406ceb37d
estimator = with_live_logs() do 
	train_posterior(simulations.θ, simulations.Z, split; seed="apple50k") 
end

# ╔═╡ 2d431f5f-88c4-4c91-9860-9948d7242c8c
sbc_result = sbc(estimator, simulations.θ, simulations.Z, split)

# ╔═╡ 0b517b7d-06a2-41d4-9eb0-29189ef21fdf
sbc_plot(sbc_result, simulations)

# ╔═╡ a3147353-d84e-4925-9be1-93ad53708754
shrinkage(estimator, simulations.θ, simulations.Z, split)

# ╔═╡ 49fd6fa6-c11d-447a-ae70-d5ba45e45811
S = posterior(estimator, flatten(x), simulations.θ[:, split.train])

# ╔═╡ a5d0d3f2-209d-427d-93cd-c8d312ceeb72
sample_posterior(S)

# ╔═╡ c9cc99b1-b43b-4f2d-bd89-ffd4e39bf30d
ppc_draws = let n = 30
	map(1:n) do i
		θ = S[:, rand(1:size(S, 2))]
		observation(run(schedule!, prior_setup.decode(θ); seed = "ppc$i"))
	end
end

# ╔═╡ 329d7c85-d751-4711-b7da-5848c97cb152
PlutoUI.ExperimentalLayout.vbox([plot_ppc_lines(c, ppc_draws, x[c]) for c in conditions])

# ╔═╡ 13ede3c6-01b0-46d3-a629-8ea199f2dc78
edge_groups = let n = length(genes)
	slot(t, f) = Symbol("$(genes[t]).edge.$(genes[f]).sgrna_cas9")
	idx(ps) = [findfirst(==(slot(t, f)), simulations.names) for (t, f) in ps]
	(; ring    = idx([(i, mod1(i - 1, n)) for i in 1:n]),
	   reverse = idx([(i, mod1(i + 1, n)) for i in 1:n]),
	   self    = idx([(i, i) for i in 1:n]))
end

# ╔═╡ 7aac6942-66bd-4213-b961-da788388ebce
function plot_marginals(S; nbins = 40, s_off = 0.1, truth=nothing)
	d = size(S, 1)
	edges = range(-1, 1; length = nbins + 1)
	role = fill(:self, d); role[edge_groups.ring] .= :ring; role[edge_groups.reverse] .= :reverse
	col = (ring = (:steelblue, 0.75), reverse = (:indianred, 0.6), self = (:grey, 0.5))
	fig = Figure(size = (680, 600))
	for k in 1:d
		ax = Axis(fig[fld1(k, 3), mod1(k, 3)];
			  title = edge_label(prior_setup.names[k]),
			  xlabel = fld1(k, 3) == 3 ? "s" : "",
			  ylabel = mod1(k, 3) == 1 ? "density" : "")


		hist!(ax, S[k, :]; bins = edges, normalization = :pdf, color = (col[role[k]], 0.75))
		hlines!(ax, [0.5]; color = :black, linestyle = :dash)
		vlines!(ax, [-s_off, s_off]; color = :grey, linestyle = :dot)
		truth === nothing || vlines!(ax, [truth[k]]; color = :crimson, linewidth = 2.5)
		xlims!(ax, -1.02, 1.02)
	end
	fig
end

# ╔═╡ fc8143b8-2576-4c3d-b444-2c7c8d9b73f7
plot_marginals(S)

# ╔═╡ a6de88cc-f0f4-42a3-a566-3adfa022eb25
let 
	R = S[edge_groups.ring, :]
	lbl = edge_label.(prior_setup.names[edge_groups.ring])
	fig = Figure(size = (680, 240))
	for (k, (i, j)) in enumerate(((1,2), (1,3), (2,3)))
		ax = Axis(fig[1, k]; xlabel = lbl[i], ylabel = lbl[j])
		datashader!(ax, Point2f.(R[i, :], R[j, :]))
	end
	fig
end

# ╔═╡ 7c603ad7-6257-461c-811e-a4e5d4b39479
let R = S[edge_groups.ring, :], n = 24,
	lbl = edge_label.(prior_setup.names[edge_groups.ring])
	bin(v) = clamp(floor(Int, (v + 1) / 2 * n) + 1, 1, n)
	H = zeros(Float32, n, n, n)
	for k in axes(R, 2)
		H[bin(R[1,k]), bin(R[2,k]), bin(R[3,k])] += 1
	end
	G = [sum(@view H[max(i-1,1):min(i+1,n), max(j-1,1):min(j+1,n), max(k-1,1):min(k+1,n)])
		 for i in 1:n, j in 1:n, k in 1:n]
	G ./= maximum(G)

	fig = Figure(size = (760, 700))
	ax = Axis3(fig[1,1]; xlabel = lbl[1], ylabel = lbl[2], zlabel = lbl[3],
			   aspect = :data, perspectiveness = 0.5, clip = false)
	contour!(ax, (-1, 1), (-1, 1), (-1, 1), G; levels = [0.08, 0.25, 0.6],
			 alpha = 0.18, colormap = :viridis, colorrange = (0, 1))
	Makie.deactivate_interaction!(ax, :scrollzoom)
	limits!(ax, -1, 1, -1, 1, -1, 1)
	fig
end;

# ╔═╡ 105f4987-9a2e-4bd2-a56c-c817cf158acd
function topology_stats(S)
	q = (1 - prior_setup.s_off) / 2
	repressive = S .< -prior_setup.s_off
	activating = S .>  prior_setup.s_off
	present    = abs.(S) .>= prior_setup.s_off
	other      = setdiff(axes(S, 1), edge_groups.ring)

	in_ring    = vec(all(repressive[edge_groups.ring, :]; dims = 1))
	in_reverse = vec(all(activating[edge_groups.reverse, :]; dims = 1))

	(; p_ring    = mean(in_ring),    bayes_factor_ring    = bayes_factor(mean(in_ring), q^3),
	   p_reverse = mean(in_reverse), bayes_factor_reverse = bayes_factor(mean(in_reverse), q^3))
end

# ╔═╡ 6d7c11e6-de39-4bbf-8b7f-ce1b11f060b8
topology_stats(S)

# ╔═╡ 58be258c-993d-4517-97d9-1d59873a1a2c
estimator_wt = with_live_logs() do
	train_posterior(simulations.θ, 
					mapslices(normalise, 
							  simulations.Z[1:length(traces[:closed].t), :, :]; 
							  dims = (1, 2)), split; seed = "applewt50k")
end

# ╔═╡ c3812b35-bb5e-4351-8de2-ff2da7894706
S_wt = posterior(estimator_wt, x_wt, simulations.θ);

# ╔═╡ 2ef3d076-1e35-4f55-9ae2-de0f94d2de64
plot_marginals(S_wt)

# ╔═╡ 46dddcc3-1a86-4878-9f7b-73dd649c459e
sample_posterior(S_wt)

# ╔═╡ 8d0c7db7-ca65-4214-bed0-1c2a60cb7b21
ppc_draws_wt = let n = 30
	map(1:n) do i
		θ = S_wt[:, rand(1:size(S_wt, 2))]
		observation(run(schedule!, prior_setup.decode(θ); seed = "ppc$i"))
	end
end

# ╔═╡ f2fa4d4c-af3e-4a78-9900-53b395cf6021
PlutoUI.ExperimentalLayout.vbox([plot_ppc_lines(c, ppc_draws_wt, x[c]) for c in conditions])

# ╔═╡ 899a6e33-604e-4408-99fb-7dc1f43de192
topology_stats(S_wt)

# ╔═╡ bf2b786f-a7bc-4b4b-b66d-3317fe8909e6
let
    Z_wt = mapslices(normalise,
                     simulations.Z[1:length(traces[:closed].t), :, :];
                     dims = (1, 2))
    
    sbc_wt = sbc(estimator_wt, simulations.θ, Z_wt, split)
    sbc_plot(sbc_wt, simulations)
end

# ╔═╡ 1651a512-a28a-40a2-82bc-c6bce628b6c3
Z_wt = mapslices(normalise,
                 simulations.Z[1:length(traces[:closed].t), :, :];
                 dims = (1, 2));

# ╔═╡ 0255c100-e787-4096-af1c-608ecb0443f3
shrinkage(estimator_wt, simulations.θ, Z_wt, split)

# ╔═╡ 7363839a-0978-4f88-87af-7230f5ea8988
S_synthetic = let
	s = zeros(length(prior_setup.names)); s[edge_groups.ring] .= -0.8
	o = observation(run(schedule!, prior_setup.decode(s); seed = "ring1"))
	posterior(estimator, flatten(o), simulations.θ[:, split.train])
end

# ╔═╡ 162434de-e304-449f-a22f-0e794c83c57b
plot_marginals(S_synthetic; truth = let t = zeros(length(prior_setup.names))
	t[edge_groups.ring] .= -0.8; t
end)

# ╔═╡ 8299b0f0-cb08-4803-aa9e-20d154ced117
topology_stats(S_synthetic)

# ╔═╡ Cell order:
# ╟─f030a2b8-7aa1-49d0-a438-d46d06870998
# ╟─90d0728c-27a2-4e10-9a32-9fc4e2f2c7fd
# ╟─49c565c1-1d38-4829-96b7-e8df709d9fd8
# ╠═7104d3da-b768-11f1-911d-25e6d0f83c37
# ╟─45c6866d-9c1c-473e-a50a-dfa07c122b23
# ╟─c8e2bf3a-4b48-459c-b1db-a426b825b237
# ╠═08c515a4-0ae5-4a7e-849f-70d204a84564
# ╟─258ad098-5c2d-47da-8a92-8b0a0ca626f2
# ╟─1d832483-b6c6-46e5-937a-5bb8bae6446e
# ╟─14914584-7bbd-47d0-a9ba-bde30464ceef
# ╠═372c40d6-fd81-4bc4-8e27-adb495123e1f
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
# ╟─3394482e-6652-4736-81ea-5a5b6daa82f8
# ╟─f5977006-3f4e-4204-9565-b4217d41f38a
# ╠═7ed69734-84a1-4e94-9a46-7d98c970d087
# ╟─57e7638a-3660-4c13-9e83-f833e9e98ed7
# ╠═491c418e-c7bc-431b-8ce3-0512932b1784
# ╠═489adcea-561c-45ae-8a35-b7aa4435cd6e
# ╠═c3a4b23f-d7b3-471c-98cc-760cd90919d2
# ╠═4aa90a45-6339-42aa-9d59-90394f8fd4e3
# ╟─cf3a114a-c147-40cf-9709-0ab901e74cae
# ╠═f1b39cf9-8f3e-4e62-a64b-7f7d8dddb661
# ╟─c2ba1b94-34d8-4e16-a799-4a4177bea7ab
# ╟─3375089b-595a-4258-b50c-c6812b604587
# ╟─d9349b15-5d62-478e-92de-d672cbf86f18
# ╟─ede66989-a2c4-43c2-8640-5944075d43e6
# ╟─b20ad086-2b04-4cc2-8d08-4dbc3fad222b
# ╟─d8198379-3f20-48ae-8ccd-a03ab4bcd714
# ╠═620088a8-1700-4938-9bc7-a647a3beabf9
# ╟─8c1554a1-1638-47f1-adfb-23c383923d95
# ╟─c508c880-7466-4eaa-9dbe-c0c1aee95da2
# ╟─e91572ab-6858-4202-854e-10db4fa0e046
# ╟─a6d2d008-93cb-4a8e-b041-f9561f39c39b
# ╠═c19bd5f5-2462-4840-9a92-e096c68dd5a6
# ╟─452a3724-e706-4881-8f7a-4c0083e80c76
# ╠═58274dc3-1edb-47af-b2fb-09585df8b474
# ╠═3872327a-c1ce-4923-8dbd-758d31f73e5e
# ╠═6e977053-1104-48eb-ac30-ae59237782c0
# ╠═66af873c-7d69-45e0-9f62-bb4b5669804c
# ╟─f2a19c0f-c339-44bf-9773-98df6ccce7c1
# ╟─8a3d393c-1bae-4179-91c5-fdc2083e43f9
# ╠═6b5dc235-83af-4584-93b4-fbea77b50674
# ╠═e540af28-9eb4-439a-9e98-800552a89aed
# ╠═d3265f2f-a2d4-452a-a284-4510808440c2
# ╟─df2878c1-1b00-4c1a-896d-a205dcd4cdad
# ╠═6d5f497d-da8b-49ea-b3d3-ef0c8d192a03
# ╠═3e493391-0720-4ecc-94c6-a93a03cfdb1f
# ╟─d641081b-f7ce-475e-b0e4-c1422403b51f
# ╟─7d515807-3dc0-43af-9341-e971d6f270a9
# ╠═0ceab4f6-7249-4dc9-a1aa-604406ceb37d
# ╟─4e692e0d-d6ba-44a4-aeed-135914a56a84
# ╠═8b29431d-bd0d-468c-8e99-4cf85c2cd945
# ╠═2d431f5f-88c4-4c91-9860-9948d7242c8c
# ╟─aef9cd8f-5a72-445f-9cfd-14ab2959df8f
# ╠═0b517b7d-06a2-41d4-9eb0-29189ef21fdf
# ╟─237184cb-50cd-40ed-ad78-9c8bbc969804
# ╟─c0cb2a78-a875-4416-b7b6-3dd70eaaab45
# ╠═a3147353-d84e-4925-9be1-93ad53708754
# ╟─d7610be6-5d7c-427a-95b1-82d903ac4282
# ╠═461d24fe-3903-4889-89df-1d451b94d1db
# ╠═49fd6fa6-c11d-447a-ae70-d5ba45e45811
# ╟─9621d07a-7fb9-4e32-863e-b57a9e9b8a11
# ╟─5edca0b9-0924-4da5-a159-12911761a06b
# ╠═a5d0d3f2-209d-427d-93cd-c8d312ceeb72
# ╟─c9cc99b1-b43b-4f2d-bd89-ffd4e39bf30d
# ╟─4abd251a-1de6-4750-be13-6067cc7e2e92
# ╠═329d7c85-d751-4711-b7da-5848c97cb152
# ╟─409aa215-cc29-4469-bf6e-f6a59e8fb844
# ╠═13ede3c6-01b0-46d3-a629-8ea199f2dc78
# ╟─08c0dbaf-a345-44a1-adc6-16e7f08fbb26
# ╟─7aac6942-66bd-4213-b961-da788388ebce
# ╟─33a08a96-7161-430e-b89a-29fb6fa6087e
# ╠═fc8143b8-2576-4c3d-b444-2c7c8d9b73f7
# ╟─a6de88cc-f0f4-42a3-a566-3adfa022eb25
# ╟─7c603ad7-6257-461c-811e-a4e5d4b39479
# ╟─03c97a0a-e488-4026-b4a5-c6fc6aa55bb4
# ╟─a56d437c-7605-4e7e-b78c-19e17f8102f8
# ╠═734acf9f-030f-49bd-8c47-55580e4421e9
# ╟─105f4987-9a2e-4bd2-a56c-c817cf158acd
# ╠═6d7c11e6-de39-4bbf-8b7f-ce1b11f060b8
# ╟─91fff83d-ff9d-4dbf-a3e8-04d14771ef1b
# ╠═58be258c-993d-4517-97d9-1d59873a1a2c
# ╠═96560e70-0ddc-4e1c-ad6c-f63602e99993
# ╠═c3812b35-bb5e-4351-8de2-ff2da7894706
# ╟─bf2b786f-a7bc-4b4b-b66d-3317fe8909e6
# ╠═1651a512-a28a-40a2-82bc-c6bce628b6c3
# ╠═0255c100-e787-4096-af1c-608ecb0443f3
# ╠═2ef3d076-1e35-4f55-9ae2-de0f94d2de64
# ╠═46dddcc3-1a86-4878-9f7b-73dd649c459e
# ╟─8d0c7db7-ca65-4214-bed0-1c2a60cb7b21
# ╟─f2fa4d4c-af3e-4a78-9900-53b395cf6021
# ╠═899a6e33-604e-4408-99fb-7dc1f43de192
# ╟─fd2da864-6c6b-4674-95f4-f56b4424d2d7
# ╠═7363839a-0978-4f88-87af-7230f5ea8988
# ╠═162434de-e304-449f-a22f-0e794c83c57b
# ╠═8299b0f0-cb08-4803-aa9e-20d154ced117
# ╟─1653db04-b38a-4f1a-857b-3b79f2e08584
# ╟─a1b50041-e44c-4acb-96fb-e433ab21a560
# ╟─1d06f01f-9fa2-4a7f-bb22-fe846cd7bff3
# ╠═93e6f519-3599-46b8-af04-d4a6eb0dd767
# ╟─94ac974b-346d-4722-859c-281dbbbf1064
