#This script calculates the squeezing for different quadratures.
using LinearAlgebra
using Plots; pyplot()
using NPZ
using LaTeXStrings

include("MagnusExpansion.jl")
include("boson.jl")
include("HarmonicCorrectionBoson.jl")

"""
Calculate the uncertainties on the x(θ) and y(θ) quadratures for
a range imax values of θ in the interval [θi, θi+θrange].
"""
function quadrature_scan(moments, imax; θi=-pi/4, θrange=pi/2)
	θ_vec, Δx2_vec, Δy2_vec = zeros(imax), zeros(imax), zeros(imax)
	for i = 1: imax
		θ = θi + (i - 1)*θrange/(imax - 1)
		θ_vec[i] = θ
		Δx2_vec[i] = (moments[3] + cos(2*θ)*moments[1] + sin(2*θ)*moments[2])/2
		Δy2_vec[i] = (moments[3] - cos(2*θ)*moments[1] - sin(2*θ)*moments[2])/2
	end
	return θ_vec, Δx2_vec, Δy2_vec
end
"""
Apply the function quadrature_scan a couple of times, always around the value of θ
where maximum squeezing has been found.
"""
function fine_quadrature_scan(moments, imax)
	θ_vec, Δx2_vec, Δy2_vec = quadrature_scan(moments, imax)
	for i = 1: 4
		k = argmin(Δy2_vec)
		θi = θ_vec[k - 1]
		θrange = θ_vec[k + 1] - θ_vec[k - 1]
		output = quadrature_scan(moments, imax; θi=θi, θrange=θrange)
		θ_vec .= output[1]
		Δx2_vec .= output[2]
		Δy2_vec .= output[3]
	end
	return θ_vec, Δx2_vec, Δy2_vec
end

#array with different gate times tf
tf0, tf1 = 2.0, 15.0
nt = 200
tf_vec = zeros(nt)
nt = size(tf_vec)[1]
dB_vec = zeros(nt, 3)
moments = zeros(nt, 3)
coeffs_array = zeros(nt, 3)

imax = 10
θopt = zeros(nt,2)

for j = 1: nt
	println(j)
	tf = tf0*(tf1/tf0)^((j - 1)/(nt - 1))
	tf_vec[j] = tf
	κ = 0.0
	# solution in the absence of counter-rotating terms
	cr_terms = false
	moments[j, :] = solve_2moments(tf, κ, cr_terms, zeros(3))
	Δx2 = (moments[j, 3] + moments[j, 1])/2
	Δy2 = (moments[j, 3] - moments[j, 1])/2
	dB_vec[j, 1] = -10.0*log10(2*Δy2)

	# solution in the presence of counter-rotating terms
	cr_terms = true
	moments[j, :] = solve_2moments(tf, κ, cr_terms, zeros(3))
	Δx2 = (moments[j, 3] + moments[j, 1])/2
	Δy2 = (moments[j, 3] - moments[j, 1])/2
	dB_vec[j, 2] = -10.0*log10(2*Δy2)

	θ_vec, Δx2_vec, Δy2_vec = fine_quadrature_scan(moments[j, :], imax)
	k = argmin(Δy2_vec)
	θopt[j, 1] = θ_vec[k]

	# solution in the presence of counter-rotating terms and correction.
	cr_terms = true
	M = correction_matrix(tf)
	order = 6
	coeffs = get_coeffs(M, order, tf)

	coeffs_array[j, :] = coeffs
	moments[j, :] = solve_2moments(tf, κ, cr_terms, coeffs)
	Δx2 = (moments[j, 3] + moments[j, 1])/2
	Δy2 = (moments[j, 3] - moments[j, 1])/2
	dB_vec[j, 3] = -10.0*log10(2*Δy2)

	θ_vec, Δx2_vec, Δy2_vec = fine_quadrature_scan(moments[j, :], imax)
	k = argmin(Δy2_vec)
	θopt[j, 2] = θ_vec[k]
end
# save figures
plot(tf_vec, dB_vec,
	 linewidth = 2,
	 labels = ["No correction" "6th order\nMagnus" "RWA"])
plot!(tickfont = font(12, "Serif"),
	  xlabel = "Pulse width "*L"\omega_\mathrm{a} t_\mathrm{f}",
	  ylabel = "Squeezing (dB)",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
savefig("squeezing.pdf")

y = hcat(θopt, zeros(size(θopt)[1]))
plot(tf_vec, y,
	 linewidth = 2,
	 labels = ["No correction" "6th order\nMagnus" "RWA"])
plot!(tickfont = font(12, "Serif"),
	  xlabel = "Pulse width "*L"\omega_\mathrm{a} t_\mathrm{f}",
	  ylabel = L"\varphi",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
savefig("squeezing_angle.pdf")

y = hcat(1.0./tf_vec, abs.(coeffs_array))
plot(tf_vec, y,
	 linewidth = 2,
	 labels = [L"1/t_\mathrm{f}" L"|c_{x,1}|" L"|c_{y,1}|" L"|\Delta|"])
plot!(yaxis = :log,
	  tickfont = font(12, "Serif"),
	  xlabel = "Pulse width "*L"\omega_\mathrm{a} t_\mathrm{f}",
	  ylabel = "coeff."*L"\times \omega_\mathrm{a}^{-1}",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
savefig("coeffs.pdf")

# save data files
output = zeros(nt,4)
output[:, 1] = tf_vec
output[:, 2: end] = dB_vec
npzwrite("dB.npy", output)
output = zeros(nt,3)
output[:, 1] = tf_vec
output[:, 2: end] = θopt
npzwrite("angle.npy", output)
output = zeros(nt,4)
output[:, 1] = tf_vec
output[:, 2: end] = coeffs_array
npzwrite("coeffs.npy", output)
