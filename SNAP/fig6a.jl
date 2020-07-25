using LinearAlgebra
using StaticArrays
using Plots; pyplot()
using FastClosures
using NPZ
using LaTeXStrings

include("MagnusExpansion.jl")
include("AuxFunctions.jl")
include("HarmonicCorrectionSnap.jl")
include("snap.jl")

function run(tf)
	σ = pauli_matrices()
	χ_GHz = (-8281.0, 0.0, 0.0)
	χ = χ_GHz./abs(χ_GHz[1])
	N = 10
	levels = SA[0, 4]
	α = (pi, pi)./2
	Ncorr = 10

	system_params = Dict("chi"=>χ, "N"=>N, "levels"=>levels, "alpha"=>α,
						 "tf"=>tf, "Ncorr"=>Ncorr)

	Identity = Matrix{Complex{Float64}}(I, 2*N, 2*N)
	Uideal = U(system_params, σ; ideal_dynamics=true, matrix=true)
	Ur = U(system_params, σ; ideal_dynamics=false, matrix=true)
	F0 = fidelity(Ur, Uideal, 2*N)
	# "First" order correction
	magnus_order = 1
	max_order = 2
	coeffs = get_coeffs(magnus_order, max_order, Ncorr, system_params)

	Ucorr = U(system_params, σ, coeffs, max_order; frame="qubit", matrix=true)
	F1 = fidelity(Ucorr, Uideal, 2*N)
	# "Second" order correction
	magnus_order = 2
	coeffs2 = get_coeffs(coeffs, magnus_order, max_order, Ncorr, system_params)

	coeffs2 = coeffs + coeffs2
	Ucorr = U(system_params, σ, coeffs2, max_order; frame="qubit", matrix=true)
	F2 = fidelity(Ucorr, Uideal, 2*N)

	return F0, F1, F2
end

npoints = 100
tf0, tf1 = 15, 200
r = (tf1/tf0)^(1/(npoints-1))
tf_array = tf0*r.^collect(0: npoints-1)
data = zeros(npoints, 4)
for i = 1: npoints
	tf = tf_array[i]
	println(tf)
	data[i, 1] = tf
	data[i, 2: end] .= run(tf)
	println(data[i, 2: end])
end

x = data[:,1]
y = log10.(1 .- data[:, [2, 3, 4]])
plot(x, y,
	 linewidth = 2,
	 labels = ["No\ncorrection" "2nd order\nMagnus" "4nd order\nMagnus"])
plot!(tickfont = font(12, "Serif"),
      xlabel = "gate time "*L"\chi t_\mathrm{f}",
      ylabel = "Fidelity error "*L"\varepsilon",
      guidefontsize = 14,
      legendfontsize = 12,
	  background_color_legend = false,
      margin = 5Plots.mm)
savefig("fidelity_error.pdf")
npzwrite("data.npy", data)
