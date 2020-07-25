using LinearAlgebra
using StaticArrays
using Plots; pyplot()
using FFTW
using FastClosures
using NPZ
using LaTeXStrings

include("/Users/tfiguei/Work/Magnus/MagnusExpansion.jl")
include("AuxFunctions.jl")
include("HarmonicCorrectionSnap.jl")
include("snap.jl")

σ = pauli_matrices()
χ_GHz = (-8281.0, 0.0, 0.0)#(-8281.0, 48.8, 0.5)
χ = χ_GHz./abs(χ_GHz[1])
N = 10
levels = SA[0, 4]#collect(0:1:9)
α = (pi/2, pi/3)#pi*(rand(10) .- 0.5)
tf = 50.0
Ncorr = 10

system_params = Dict("chi"=>χ, "N"=>N, "levels"=>levels, "alpha"=>α,
					 "tf"=>tf, "Ncorr"=>Ncorr)

UI = U(system_params, σ; ideal_dynamics=false, frame="interaction", matrix=true)
Identity = Matrix{Complex{Float64}}(I, 2*N, 2*N)
println(1 - fidelity(UI, Identity, 2*N))

Uideal = U(system_params, σ; ideal_dynamics=true, matrix=true)
Ur = U(system_params, σ; ideal_dynamics=false, matrix=true)
println(1 - fidelity(Ur, Uideal, 2*N))

# "First" order correction
magnus_order = 1
max_order = 2
coeffs = get_coeffs(magnus_order, max_order, Ncorr, system_params)
println("---")
Ucorr = U(system_params, σ, coeffs, max_order; frame="qubit", matrix=true)
println(1 - fidelity(Ucorr, Uideal, 2*N))
# "Second" order correction
magnus_order = 2
coeffs2 = get_coeffs(coeffs, magnus_order, max_order, Ncorr, system_params)
println("---")
coeffs2 = coeffs + coeffs2
Ucorr = U(system_params, σ, coeffs2, max_order; frame="qubit", matrix=true)
println(1 - fidelity(Ucorr, Uideal, 2*N))

# In the lab frame
println("Results in the lab frame")
Uideal = U(system_params, σ; frame="lab", ideal_dynamics=true, matrix=true)
Unideal = U(system_params, σ; frame="lab", ideal_dynamics=false, matrix=true)
Ucorr = U(system_params, σ, coeffs2, max_order; frame="lab", matrix=true)
println(1 - fidelity(Unideal, Uideal, 2*N))
println(1 - fidelity(Ucorr, Uideal, 2*N))

#FFT results
Npoints = 1000
data = zeros(Npoints, 5)
for i = 1: Npoints
	t = (i-1)*tf/(Npoints-1)
	data[i,1] = t
	data[i, 2: 3] .= pulse_envelope(t, tf, χ, levels, α)
	data[i, 4: 5] .= data[i, 2: 3]
	data[i, 4: 5] .+= correction_pulse_envelope(t, tf, χ, coeffs2, max_order, Ncorr)
end

x = data[:, 1]
y = data[:, [2, 4]]
p1 = plot(x, y,
	 linewidth = 2,
	 labels = ["Initial\npulse" "Corrected\npulse"],
	 layout = (2, 1),
	 title = ["x quadrature" " "])
plot!(xlim=(0, tf),
	  tickfont = font(12, "Serif"),
	  xlabel = "time "*L"t/t_\mathrm{f}",
	  ylabel = "pulse "*L"\times |\chi|^{-1}",
	  guidefontsize = 14,
	  legendfontsize = 12,
	  margin = 5Plots.mm)

x = data[:, 1]
y = data[:, [3, 5]]
p2 = plot(x, y,
	 linewidth = 2,
	 labels = ["Initial\npulse" "Corrected\npulse"],
	 layout = (2, 1),
	 title = ["y quadrature" " "])
plot!(xlim=(0, tf),
	  tickfont = font(12, "Serif"),
	  xlabel = "time "*L"t/t_\mathrm{f}",
	  ylabel = "pulse "*L"\times |\chi|^{-1}",
	  guidefontsize = 14,
	  legendfontsize = 12,
	  background_color_legend = false,
	  margin = 5Plots.mm)
plot(p1, p2, layout=(1,2), size = (800, 800))
savefig("pulse.pdf")
# Add padding to the pulse
data = vcat(zeros(Npoints, 5), data, zeros(Npoints, 5))

fft_data = zeros(size(data))
fft_data[:, 1] = fftfreq(size(data)[1], tf/(Npoints-1))
for i = 2: 5
	fft_pulse = fft(data[:, i])
	fft_data[:, i] = abs.(fft_pulse)
end

a = div(size(data)[1], 2)
x = fft_data[1: a, 1]
y = fft_data[1: a, [2, 4]]
p1 = plot(x, y,
		  linewidth = 2,
		  labels = ["Initial\npulse" "Corrected\npulse"],
		  linestyle = [:solid :dash])
plot!(xlim=(0, 10),
	  tickfont = font(12, "Serif"),
	  xlabel = "Frequency "*L"\omega/\chi",
	  ylabel = L"|S_x(x)|",
	  guidefontsize = 14,
	  legendfontsize = 12,
	  background_color_legend = false,
	  margin = 5Plots.mm)

x = fft_data[1: a, 1]
y = fft_data[1: a, [3, 5]]
p2 = plot(x, y,
	 linewidth = 2,
	 labels = ["Initial\npulse" "Corrected\npulse"],
	 linestyle = [:solid :dash])
plot!(xlim=(0, 10),
	  tickfont = font(12, "Serif"),
	  xlabel = "Frequency "*L"\omega/\chi",
	  ylabel = L"|S_y(x)|",
	  guidefontsize = 14,
	  legendfontsize = 12,
	  background_color_legend = false,
	  margin = 5Plots.mm)

plot(p1, p2, layout=(1,2), size = (800, 400))

npzwrite("fft.npy", fft_data[1: div(Npoints,2), :])
savefig("fft.pdf")
