#This script calculates the harmonic correction for a 3-level system.
#Here npulse is fixed.
using DifferentialEquations
using LinearAlgebra
using Plots; pyplot()
using NPZ
using LaTeXStrings

include("MagnusExpansion.jl")
include("HarmonicCorrection3ls.jl")
include("3ls.jl")
include("Envelopes.jl")

function save_pulse(α, tf, coeffs, pulse, basis)
	t_array = collect(0: tf/100: tf)
	output = zeros(length(t_array), 6)
	for i = 1: length(t_array)
		t = t_array[i]
		output[i, 1] = abs(α)*t
		output[i, 2] = Δ(t, tf, coeffs[1], basis.detuning)./abs(α)
		f = pulse(t)/abs(α)
		output[i, 3] = f
		g = g_envelopes(t, tf, coeffs[2], basis.pulse)./abs(α)
		output[i, 5] = (f + g[1])
		output[i, 6] = g[2]
	end
	plot(output[:, 1], output[:, [3, 2, 5, 6]],
		 linewidth = 2,
		 labels = [L"f_x^{(0)}(t)" L"\Sigma \Delta^{(6)}" L"f_x^{(6)}(t)" L"f_y^{(6)}(t)"],
		 linestyle = [:solid :dash :dash :dash])
	plot!(tickfont = font(12, "Serif"),
		  xlabel = "time "*L"|\alpha| t",
		  ylabel = "Envelopes "*L"\times |\alpha|^{-1}",
		  guidefont = font(14, "Serif"),
		  legendfont = font(12, "Serif"),
		  background_color_legend = false,
		  margin = 5Plots.mm)
	savefig("pulse_envelopes.pdf")
	npzwrite("pulse"*string(tf)*".npy", output)
	return nothing
end

function run_(magnus_order)
	λ = gellmann_matrices()
	α = -0.1
	η = sqrt(2.0)
	θ = pi/2
	# setting the number of harmonics of the correction basis
	nΔ = 0
	npulse = 2
	# array with different gate times tf
	tf0, tf1 = 50.0, 155.0
	nt = 100
	tf_array = collect(tf0: (tf1 - tf0)/(nt - 1): tf1)
	nt = size(tf_array)[1]
	F0, Fmagnus, Fdrag = zeros(nt), zeros(nt, length(magnus_order)), zeros(nt)

	for j = 1: nt
		tf = tf_array[j]
		println(tf)

		pulse = @closure t-> gaussian_pulse(t, tf, θ)
		pulse_integral = @closure t-> gaussian_pulse_integral(t, tf, θ)
		pulse_derivative = @closure t-> gaussian_pulse(t, tf, θ; derivative=true)
		system_params = Dict("alpha"=> α, "eta"=> η, "tf"=> tf, "pulse"=> pulse,
							 "pulse_integral"=> pulse_integral,
							 "pulse_derivative"=> pulse_derivative)
		# Ideal evolution
		U_ideal = U(system_params, λ, ideal_dynamics=true)
		# Nonideal evolution
		U_nideal = U(system_params, λ, ideal_dynamics=false)
		# Harmonic correction
		F0[j] = fidelity(U_nideal, U_ideal, 2)

		basis = fourier_basis(tf, nΔ, npulse)
		Δ_basis = basis.detuning
		pulse_basis = basis.pulse
		# correction matrix
		M = correction_matrix(system_params, Δ_basis, pulse_basis)
		# Magnus correction
		for k = 1: length(magnus_order)
			coeffs = get_coeffs(M, magnus_order[k], system_params, Δ_basis, pulse_basis)
			U_nideal2 = U(system_params, λ, coeffs, Δ_basis, pulse_basis)

			Fmagnus[j, k] = fidelity(U_nideal2, U_ideal, 2)
			if j == 1 && magnus_order[k] == 6
				save_pulse(α, tf, coeffs, pulse, basis)
			end
		end
		# Drag correction
		U_nideal3 = Udrag(system_params, λ; high_order=false)
		Fdrag[j] = fidelity(U_nideal3, U_ideal, 2)
	end
	return abs(α)*tf_array, F0, Fmagnus, Fdrag
end

magnus_order = [2, 6]
t, F0, Fmagnus, Fdrag = run_(magnus_order)
y = 1 .- hcat(F0, Fmagnus, Fdrag)
plot(t, y,
	 linewidth = 2,
	 labels = ["No\ncorrection" "2nd order\nMagnus" "6th order\nMagnus" "DRAG"],
	 linestyle = [:solid :solid :solid :dash])

plot!(yaxis=:log,
	  tickfont = font(12, "Serif"),
	  xlabel = "gate time "*L"|\alpha| t_\mathrm{f}",
	  ylabel = "Fidelity error "*L"\varepsilon",
	  guidefont = font(14, "Serif"),
	  legendfont = font(12, "Serif"),
	  background_color_legend = false,
	  margin = 5Plots.mm)
savefig("fidelity_error.pdf")

data = hcat(t, F0, Fmagnus, Fdrag)
npzwrite("data.npy", data)
# data[:, 1] .= t
# data[:, 2] .= F0
# data[:, 3] .= Fmagnus
# for m in magnus_order
# if m == 1
# 	npzwrite("fidelity_1st.npy", data)
# elseif m == 2
# 	npzwrite("fidelity_2nd.npy", data)
# else
# 	npzwrite("fidelity_"*string(m)*"th.npy", data)
# end

# data = zeros(nt,2)
# data[:,1] = abs(α)*tf_array
# data[:,2] = Fdrag
# npzwrite("fidelity_drag.npy", data)
