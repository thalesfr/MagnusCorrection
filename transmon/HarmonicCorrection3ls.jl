import DifferentialEquations;	const DE = DifferentialEquations
using LinearAlgebra
using StaticArrays
using Combinatorics
using FastClosures

"""
	g(t, tf, coeffs, basis_functions)
Correction pulse
# Arguments
- `t::Real or t::Array{<:Real,1}`: time.
- `tf::Real`: total time of the gate.
- `coeffs::Array{<:Real, 1}`: coefficients of the correction pulse.
- `basis_functions::`: .
"""
function g(t::Real, tf, coeffs, basis)
	basis_vector = basis.vector
	basis.function!(basis_vector, t)
	cost, sint = cos(t), sin(t)
	@inbounds gt = (coeffs[1]*cost + coeffs[2]*sint)*basis_vector[1]
	for i = 2: length(basis_vector)
		@inbounds gt += (coeffs[2*i - 1]*cost + coeffs[2*i]*sint)*basis_vector[i]
	end
	return gt
end

function g_envelopes(t, tf, coeffs, basis)
	basis_vector = basis.vector
	basis.function!(basis_vector, t)
	gt = zeros(2)
	for i = 1: length(basis_vector)
		gt[1] += coeffs[2*i - 1]*basis_vector[i]
		gt[2] += coeffs[2*i]*basis_vector[i]
	end
	return gt
end

"""
	Δ(t, tf, coeffs, basis_functions)
Detuning of the pulse frequency with respect to the transition frequency between the zero'th
and first levels.
# Arguments
- `t::Real`: time.
- `tf::Real`: total time of the gate.
- `coeffs::Array{<:Real, 1}`: coefficients of the detuning.
- `basis_functions::`: .
"""
function Δ(t::Real, tf, coeffs, basis)
	basis_vector = basis.vector
	basis.function!(basis_vector, t)
	@inbounds Δt = coeffs[1]*basis_vector[1]
	for i = 2: length(basis_vector)
		@inbounds Δt += coeffs[i]*basis_vector[i]
	end
	return Δt
end

"""
	Wharmonic_I(t, tf, α, η, coeffs, Δ_basis, pulse_basis)
The harmonic correction Hamiltonian in the interaction picture.
# Arguments
- `t::Real`: time.
- `tf::Real`: total time of the gate.
- `α::Real`: the anarmonicity in units of the transition frequency between the zeroth
and first energy levels.
- `η::Real`: parameter that characterizes the strength of the 1<->2 transition relative
to the 0<->1 transition.
- `coeffs::Array{Array{<:Real, 1}, 1}`: The first and second elements are the arrays with the
Δ coefficients and pulse coefficients, respectively.
- `Δ_basis::`: .
- `pulse_basis::`: .
"""
function Wharmonic_I(t, tf, α, η, c, coeffs, Δ_basis, pulse_basis)
	ct = c(t)
	cosc, sinc = cos(ct), sin(ct)
	cos2c, sin2c = cos(2*ct), sin(2*ct)
	φ = (1 + α)*t
	cosφ, sinφ = cos(φ), sin(φ)
	sint = sin(t)

	Δt = Δ(t, tf, coeffs[1], Δ_basis)/2
	gt = g(t, tf, coeffs[2], pulse_basis)
	ηg = η*gt

	WIt = @SVector [gt*cos(t),
				    -gt*sint*cos2c + Δt*sin2c,
				    gt*sint*sin2c + Δt*cos2c,
				    ηg*cosφ*cosc,
				    ηg*sinφ*cosc,
				    ηg*sinφ*sinc,
				    -ηg*cosφ*sinc,
				    -sqrt(3.0)*Δt,
				    Δt]

	return WIt
end
"""
	Wharmonic(t, tf, α, η, coeffs, Δ_basis, pulse_basis)
The harmonic correction Hamiltonian.
# Arguments
- `t::Real`: time.
- `tf::Real`: total time of the gate.
- `α::Real`: the anarmonicity in units of the transition frequency between the zeroth
and first energy levels.
- `η::Real`: parameter that characterizes the strength of the 1<->2 transition relative
to the 0<->1 transition.
- `coeffs::Array{Array{<:Real, 1}, 1}`: The first and second elements are the arrays with the
Δ coefficients and pulse coefficients, respectively.
- `Δ_basis::Function`: Detuning basis functions as a function of t.
- `pulse_basis::Function`: Pulse basis functions as a function of t.
"""
function Wharmonic(t, tf, η, coeffs, Δ_basis, pulse_basis)
	Δt = Δ(t, tf, coeffs[1], Δ_basis)/2
	gt = g(t, tf, coeffs[2], pulse_basis)
	cost, sint = cos(t), sin(t)
	ηg = η*gt

	Wt = @SVector [gt*cost,
				   -gt*sint,
				   Δt,
				   ηg*cost,
				   ηg*sint,
				   0,
				   0,
				   -sqrt(3.0)*Δt,
				   Δt]
	return Wt
end
# correction matrix obtained from the selected envelopes
function du!(du, u, args, t)
	tf = args[1]
	α = args[2]
	η = args[3]
	c = args[4]
	coeffs = args[5]
	Δ_basis = args[6]
	pulse_basis = args[7]

	W0t = Wharmonic_I(t, tf, α, η, c, coeffs, Δ_basis, pulse_basis)
	du .= view(W0t, 1: 7)
	return nothing
end
"""
	correction_matrix(tf, α, η, Δ_basis, pulse_basis; mask=nothing)
Return the correction matrix.
# Arguments
- `tf::Real`: total time of the gate.
- `α::Real`: anarmonicity.
- `η::Real`: parameter that characterizes the strength of the 1<->2 transition relative
to the 0<->1 transition.
- `coeffs::Array{Array{<:Real, 1}, 1}`: The first and second elements are the arrays with the
Δ coefficients and pulse coefficients, respectively.
- `Δ_basis::Function`: detuning basis functions as a function of t.
- `pulse_basis::Function`: pulse basis functions as a function of t.
- `mask::Array{Bool, 1}`: boolean array with a selection of the coefficients.
"""
function correction_matrix(system_params, Δ_basis, pulse_basis)
	α = system_params["alpha"]
    η = system_params["eta"]
    tf = system_params["tf"]
	pulse_integral = system_params["pulse_integral"]

	reltol, abstol = 1e-8, 1e-13
	npulse_basis = length(pulse_basis.vector)
	nΔ_basis = length(Δ_basis.vector)
	M = zeros(7, nΔ_basis + 2*npulse_basis)

	v = zeros(nΔ_basis + 2*npulse_basis)
	for i = 1: nΔ_basis + 2*npulse_basis
		v .= 0.0
		v[i] = 1.0
		coeffs = (v[1: nΔ_basis], v[nΔ_basis + 1: end])

		u0 = zeros(7)
		tspan = (zero(tf), tf)
		args = (tf, α, η, pulse_integral, coeffs, Δ_basis, pulse_basis)
		prob = DE.ODEProblem(du!, u0, tspan, args)
		sol = DE.solve(prob, DE.Vern9(), reltol=reltol, abstol=abstol, saveat=[tf])

		M[:, i] .= sol(tf)
	end
	return M
end
"""
	coefficients_W(Omega, M, Δ_basis, pulse_basis)
Return the coefficients that solve the linear system `M*coeffs = Omega`.
# Arguments
- `Omega::Array{<:Real, 1}`: the error vector, obtained via the Magnus expansion.
- `M::Array{<:Real, 2}`: correction matrix.
- `Δ_basis::Function`: detuning basis functions as a function of t.
- `pulse_basis::Function`: pulse basis functions as a function of t.
"""
function coefficients_W(Omega, M, Δ_basis, pulse_basis)
	ih_vec = Omega[1: 7]
	nΔ_basis = length(Δ_basis.vector)
	coeffs = pinv(M)*ih_vec

	return (coeffs[1: nΔ_basis], coeffs[nΔ_basis + 1: end])
end
"""
	get_coeffs(M, order, system_params, Δ_basis, pulse_basis)
Return the sum of correction coefficients up to order `order`.
# Arguments
- `M::Array{<:Real, 2}`: correction matrix.
- `order::Integer`: the order of the Magnus correction.
- `system_params::Dictionary`: dictionary with the parameters of the system.
- `Δ_basis::Function`: detuning basis functions as a function of t.
- `pulse_basis::Function`: pulse basis functions as a function of t.
"""
function get_coeffs(M, order, system_params, Δ_basis, pulse_basis; max_norm=nothing)
    nΔ_basis = length(Δ_basis.vector)
    npulse_basis = length(pulse_basis.vector)
    coeffs = (zeros(nΔ_basis), zeros(2*npulse_basis))
    for k = 1: order
        coeffs2 = get_coeffs(M, coeffs, k, system_params, Δ_basis, pulse_basis)
		#coeffs2 = get_coeffs(M, coeffs, order, system_params, Δ_basis, pulse_basis)
        coeffs = coeffs .+ coeffs2
		if max_norm != nothing
			if norm(vcat(coeffs...)) > max_norm
				return (zeros(nΔ_basis), zeros(2*npulse_basis))
			end
		end
    end
    return coeffs
end
"""
	get_coeffs(M, order, system_params, Δ_basis, pulse_basis)
Return the correction coefficients of order `order`.
# Arguments
- `M::Array{<:Real, 2}`: correction matrix.
- `coeffs::Tuple{Array{<:Real, 1}, Array{<:Real, 1}}`: correction coefficients.
- `order::Integer`: the order of the Magnus correction.
- `system_params::Dictionary`: dictionary with the parameters of the system.
- `Δ_basis::Function`: detuning basis functions as a function of t.
- `pulse_basis::Function`: pulse basis functions as a function of t.
"""
function get_coeffs(M, coeffs, order, system_params, Δ_basis, pulse_basis)
	α = system_params["alpha"]
    η = system_params["eta"]
    tf = system_params["tf"]
	pulse = system_params["pulse"]
	pulse_integral = system_params["pulse_integral"]

	A = @closure (
		t-> - V_I(t, α, η, pulse, pulse_integral)
			- Wharmonic_I(t, tf, α, η, pulse_integral, coeffs, Δ_basis, pulse_basis))

    Ndims = 9
    Omegaf = solveOmega(tf, order, A, comm!, Ndims)
    vec = sum(Omegaf, dims=2)
    coeffs = coefficients_W(vec, M, Δ_basis, pulse_basis)
    return coeffs
end

function find_best_coeffs(ncoeffs, order, system_params, Δ_basis, pulse_basis, θ, λ)
    nΔ_basis = length(Δ_basis.vector)
    npulse_basis = length(pulse_basis.vector)
    if ncoeffs > nΔ_basis + 2*npulse_basis
        error("ncoeffs cannot be larger than ", nΔ_basis + 2*npulse_basis, ".")
    end

    α = system_params["alpha"]
    η = system_params["eta"]
    tf = system_params["tf"]
    # find the correction matrix
    #M = correction_matrix(tf, α, η, Δ_basis, pulse_basis)
	M = correction_matrix(system_params, Δ_basis, pulse_basis)
    # Ideal evolution operator
    U_ideal = U(system_params, λ; ideal_dynamics=true)
    # mask vector with the selected `ncoeffs` correction coefficients.
    coeffs_mask = trues(nΔ_basis + 2*npulse_basis)
    coeffs_mask[1: ncoeffs] = falses(ncoeffs)
    # all possible permutations of `ncoeffs` coefficients
    permutations = multiset_permutations(coeffs_mask, length(coeffs_mask))
    permutations = collect(permutations)
	println("There are ", length(permutations), " possible combinations of coefficients. "*
			"Starting to calculate ...")
    # select the best subgroup of `ncoeffs` coefficients
    coeffs_opt = (zeros(nΔ_basis), zeros(2*npulse_basis))
    Fopt = 0.0
    for i = 1: length(permutations)
        # Apply mask to the matrix M and obtain coefficients of the ith permutation
        Mp = copy(M)
        Mp[:, permutations[i]] .= 0.0
        coeffs = get_coeffs(Mp, order, system_params, Δ_basis, pulse_basis;
							max_norm= 2*θ/tf)

        Ucorrected = U(system_params, λ, coeffs, Δ_basis, pulse_basis)
        Fcorrected = fidelity(Ucorrected, U_ideal, 2)
        if Fcorrected > Fopt
            Fopt = Fcorrected
            coeffs_opt = coeffs
        end
    end
    return coeffs_opt
end
"""
	U(system_params, coeffs, λ, Δ_basis, pulse_basis)
Corrected evolution operator.
# Arguments
- `system_params::Dictionary`: dictionary with the parameters of the system.
- `λ::Array{Array{<:Complex, 2}}`: Array with the Gell-Mann
- `coeffs::Tuple{Array{<:Real, 1},Array{<:Real, 1}}`: correction coefficients.
- `Δ_basis::Function`: detuning basis functions as a function of t.
- `pulse_basis::Function`: pulse basis functions as a function of t.
"""
function U(system_params, λ, coeffs, Δ_basis, pulse_basis)
    α = system_params["alpha"]
    η = system_params["eta"]
    tf = system_params["tf"]
	pulse = system_params["pulse"]

	H_corrected = @closure t-> (H0(t, α, η, pulse)
						  		+ Wharmonic(t, tf, η, coeffs, Δ_basis, pulse_basis))

    sol = solveU(H_corrected, tf, λ)
    return sol(tf)
end

function solveρ(system_params, λ, coeffs, Δ_basis, pulse_basis, c1, c2, ρ0)
	tf = system_params["tf"]
	α = system_params["alpha"]
	η = system_params["eta"]
	pulse = system_params["pulse"]

	H = @closure t-> (H0(t, α, η, pulse)
					  + Wharmonic(t, tf, η, coeffs, Δ_basis, pulse_basis))

	tspan = (zero(tf), tf)
	args = (H, c1, c2, λ)
	prob = DE.ODEProblem(dρ, ρ0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-14)
	if sol.retcode == :Success
		return sol
	else
		error("Could not solve the master equation.")
	end
end

function average_fidelity(system_params, λ, coeffs, Δ_basis, pulse_basis, c1, c2)
	tf = system_params["tf"]
	Uideal = U(system_params, λ; ideal_dynamics = true)

	dict = Dict("+x"=> (1.0, 0.0, 0.0), "-x"=> (-1.0, 0.0, 0.0),
				"+y"=> (0.0, 1.0, 0.0), "-y"=> (0.0, -1.0, 0.0),
				"+z"=> (0.0, 0.0, 1.0), "-z"=> (0.0, 0.0, -1.0))
	F = 0.0
	for tag in ("+x", "-x", "+y", "-y", "+z", "-z")
		a = dict[tag]
		ρ0 = ρbloch(a)
		ρf = solveρ(system_params, λ, coeffs, Δ_basis, pulse_basis, c1, c2, ρ0)
		ρf = ρf(tf)
		ρfideal = Uideal*ρ0*adjoint(Uideal)
		F += tr(ρfideal*ρf)/6
	end
	if imag(F)>10.0^(-10.0)
		error("Complex fidelity!")
	else
		return real(F)
	end
end
