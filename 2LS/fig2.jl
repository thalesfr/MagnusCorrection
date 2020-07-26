#This script calculates the harmonic correction for a 2-level system.
using LinearAlgebra
using Plots; pyplot()
#using Statistics
using NPZ
using LaTeXStrings

include("MagnusExpansion.jl")
include("2ls.jl")
include("HarmonicCorrection2ls.jl")

σ = pauli_matrices()
#array with different gate times tf
tf0, tf1 = 0.1, 30.0
nt = 200
tf_array = collect(tf0: (tf1-tf0)/(nt-1): tf1)
nt = length(tf_array)

hDelta1(x) = 1.0
h1(x) = 1.0-cos(2.0*pi*x)
h2(x) = sin(2.0*pi*x)
h3(x) = 1.0-cos(4.0*pi*x)
env_Delta = (hDelta1,)
env_pulse1 = (h1,)
env_pulse2 = (h1,)

ncoeffs = length(env_Delta) + length(env_pulse1) + length(env_pulse2)
coeffs_array = zeros(ncoeffs, nt)
F1, F2 = zeros(nt), zeros(nt)
for j = 1: nt
	tf = tf_array[j]
	println(tf)
	# Ideal evolution
	U_ideal = U(tf, σ; ideal_dynamics=true)
	# Nonideal evolution
	U_nideal = U(tf, σ; ideal_dynamics=false)
	# Fidelity
	F1[j] = fidelity(U_nideal, U_ideal, 2)
	# Harmonic correction
	M = correction_matrix(tf, env_Delta, env_pulse1, env_pulse2)
	order = 2
	coeffs = get_coeffs(M, order, tf, env_Delta, env_pulse1, env_pulse2)
	coeffs_array[:,j] .= vcat(coeffs[1], coeffs[2])
	# Corrected evolution
	U_nideal2 = U(tf, σ, coeffs, env_Delta, env_pulse1, env_pulse2)
	# Fidelity
	F2[j] = fidelity(U_nideal2, U_ideal, 2)
end

# aux = zeros(nt,2)
# aux[:,1] = tf_array
# aux[:,2] = log10.(1.0 .- F1)
y = 1 .- hcat(F1, F2)
plot(tf_array, y,
	 linewidth = 2,
	 labels = ["No correction" "2nd order\nMagnus"])
plot!(yaxis = :log,
	  tickfont = font(12, "Serif"),
	  xlabel = "Gate time "*L"\omega_\mathrm{q} t_\mathrm{f}",
	  ylabel = "Fidelity error "*L"\varepsilon",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
# plot(tf_array, 1.0 .- F1, xtickfont = font(16), ytickfont = font(16),
#      linewidth=3, yaxis=:log)
# plot!(tf_array, 1.0 .- F2, linewidth=3)
savefig("fidelity_2ls.pdf")

y = abs.(transpose(coeffs_array))
plot(tf_array, y,
	 linewidth = 2,
	 labels = [L"|\Delta|" L"|c_{x, 1}|" L"|c_{y, 1}|"])
plot!(yaxis = :log,
	  tickfont = font(12, "Serif"),
	  xlabel = "Gate time "*L"\omega_\mathrm{q} t_\mathrm{f}",
	  ylabel = "Coeff. "*L"\times \omega_\mathrm{q}^{-1}",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
savefig("coeffs.pdf")

output = zeros(nt, ncoeffs)
output[:, 1] = tf_array
output[:, 2] = F1
output[:, 3] = F2
npzwrite("fidelity.npy", output)
output = zeros(nt, ncoeffs + 1)
output[:, 1] = tf_array
output[:, 2: end] = transpose(coeffs_array)
npzwrite("correction_coef.npy", output)
