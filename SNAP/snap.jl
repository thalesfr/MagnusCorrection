using StaticArrays
using LinearAlgebra
import DifferentialEquations;	const DE = DifferentialEquations

function pauli_matrices(; type=Float64)
	σ1 = SMatrix{2, 2, Complex{type}}([0 1; 1 0])
	σ2 = SMatrix{2, 2, Complex{type}}([0 -1im; 1im 0])
	σ3 = SMatrix{2, 2, Complex{type}}([1 0; 0 -1])
	σ4 = SMatrix{2, 2, Complex{type}}([1 0; 0 1])
	σ = (σ1, σ2, σ3, σ4)
	return σ
end
"""
Cosine envelope function.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `n::Integer`: harmonic order.
"""
function fc(t, tf, n)
	tau = t/tf
	ft = (1 - cos(2*pi*n*tau))/tf
	return ft
end
"""
Sine envelope function.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `n::Integer`: harmonic order.
"""
function fs(t, tf, n)
	tau = t/tf
	ft = sin(2*pi*n*tau)/tf
	return ft
end
"""
Definite integral of fc.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `n::Integer`: harmonic order.
"""
function intfc(t, tf, n)
	tau = t/tf
	c = 2*pi*n
	return tau - sin(c*tau)/c
end
"""
Driving fields in the x and y direction in a given subspace n of the bosonic system.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `α::Real`: angle of the rotation axis.
"""
function f(t, tf, α)
	#ft = zeros(2)
	theta = pi/2
	T = tf/2
	if t < T
		# ft[1] = theta*fc(t, T, 1)
		fx, fy = theta*fc(t, T, 1), 0.0
	else
		fct = theta*fc(t - T, T, 1)
		# ft[1], ft[2] = fct*cos(α), fct*sin(α)
		fx, fy = fct*cos(α), fct*sin(α)
	end
	#return ft
	return fx, fy
end
"""
Driving fields in the x and y direction in the whole truncated bosonic Hilbert space.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `α::Array{Real}`: angle of the rotation axes.
- `N::Integer`: size of the truncated bosonic Hilbert space.
"""
function f(t, tf, levels, α, N)
	ft = zeros(2, N)
	theta = pi/2
	index = levels .+ 1
	T = tf/2
	if t < T
		ft[1, index] .= theta*fc(t, T, 1)
	else
		fct = theta*fc(t - T, T, 1)
		ft[1, index] .= fct.*cos.(α)
		ft[2, index] .= fct.*sin.(α)
	end
	return ft
end
"""
Integral of the driving fields in the x and y direction in in a given subspace n of the
bosonic system.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `α::Array{Real}`: angle of the rotation axes.
"""
function intf(t, tf, α)
	theta = pi/2

	intfct = theta*intfc(t, tf, 1)
	intfx = intfct*cos(α)
	intfy = -intfct*sin(α)

	return intfx, intfy
end
"""
Integral of the driving fields in the x and y direction in the whole truncated bosonic
Hilbert space.
# Arguments
- `t::Real`: time.
- `tf::Real`: duration of the snap gate.
- `levels::Array{Integer}`: bosonic levels that are resonantly driven.
- `α::Array{Real}`: angle of the rotation axes.
- `N::Integer`: size of the truncated bosonic Hilbert space.
"""
function intf(t, tf, levels, α, N)
	intft = zeros(2, N)
	theta = pi/2
	index = levels .+ 1

	intfct = theta*intfc(t, tf, 1)
	intft[1, index] .= intfct.*cos.(α)
	intft[2, index] .= -intfct.*sin.(α)

	return intft
end

function F(t, tf, χ, levels, α, N; ideal_interaction::Bool=false)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	Ft = zeros(4, N)
	for i = 1: length(levels)
		Fj!(Ft, t, tf, χef, levels[i], α[i], N, ideal_interaction)
	end
	return Ft
end

function Fj!(Ft, t, tf, χef, j, α, N, ideal_interaction)
	fx, fy = f(t, tf, α)
	if ideal_interaction == true
		Ft[1, j + 1] = fx
		Ft[2, j + 1] = -fy
	else
		for n = 0: N - 1
			dφn = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
			dφj = χef[1]*j + χef[2]*j*(j - 1) + χef[3]*j*(j - 1)*(j - 2)
			φ = (dφn - dφj)*t
			cosφ = cos(φ)
			sinφ = sin(φ)
			Ft[1, n + 1] +=  + fx*cosφ - fy*sinφ
			Ft[2, n + 1] +=  - fx*sinφ - fy*cosφ
		end
	end
	return nothing
end

function V(t, tf, χ, levels, α, N)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	Vt = zeros(4, N)
	for i = 1: length(levels)
		Vj!(Vt, t, tf, χef, levels[i], α[i], N)
	end
	return Vt
end

function Vj!(Vt, t, tf, χef, j, α, N)
	fx, fy = f(t, tf, α)
	for n = 0: N - 1
		if n != j
			dφn = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
			dφj = χef[1]*j + χef[2]*j*(j - 1) + χef[3]*j*(j - 1)*(j - 2)
			φ = (dφn - dφj)*t
			cosφ = cos(φ)
			sinφ = sin(φ)
			Vt[1, n + 1] +=  + fx*cosφ - fy*sinφ
			Vt[2, n + 1] +=  - fx*sinφ - fy*cosφ
		end
	end
	return nothing
end

function VI(t, tf, χ, levels, α, N)
	T = tf/2
	Vt = V(t, tf, χ, levels, α, N)
	v = zeros(4, N)
	if t < T + 10.0*eps()
		v[1: 2, :] .= intf(t, T, levels, α.*0, N)
		unitary_tranformation!(Vt, v, N)
	else
		v[1: 2, :] .= intf(t - T, T, levels, α, N)
		unitary_tranformation!(Vt, v, N)
		v[1: 2, :] .= intf(T, T, levels, α.*0, N)
		unitary_tranformation!(Vt, v, N)
	end
	return Vt
end

function VI2(t, tf, χ, levels, α, N)
	Vt = V(t, tf, χ, levels, α, N)
	Vt = vector2matrix(Vt, σ)
	U0 = U(t, tf, χ, levels, α, N)
	VIt = adjoint(U0)*Vt*U0
	VI2t = zeros(4, N)
	for i = 1: N
		VI2t[:, i] = pauli_decomp(VIt[2*(i - 1) + 1: 2*i, 2*(i - 1) + 1: 2*i])
	end
	return VI2t
end

function H(t, F, args)
    σ = args[1]
	N = args[2]
    Ft = F(t)
	Ht = zeros(Complex{Float64}, 2, 2, N)
	for j = 1: N
		vj = view(Ft,:,j)
		Htj = view(Ht, :, :, j)
		vector2matrix!(Htj, vj, σ)
	end
	return Ht
end

function U(t, tf, χ, levels, α, N, σ)
	index = levels .+ 1
	aux = zeros(Complex{Float64}, N, 4)
	T = tf/2
	if t < T
		aux[1:2,:] .= -intf(t, T, levels, α.*0, N)
		aux .= exp_matrix(aux, N)
		U0 = vector2matrix(aux, σ)
	else
		aux[1: 2, :] .= -intf(T, T, levels, α.*0, N)
		aux .= exp_matrix(aux, N)
		U0 = vector2matrix(aux, σ)

		aux[1: 2, :] .= -intf(t-T, T, levels, α, N)
		aux[3: 4, :] .= zero(Complex{Float64})#zeros(Complex{Float64}, 2, N)
		aux .= exp_matrix(aux, N)
		M = vector2matrix(aux, σ)
		U0 = M*U0
	end
	return U0
end

function U2(t, tf, χ, levels, α, N, σ)
	index = levels .+ 1
	aux = zeros(4, N)
	M = zeros(Complex{Float64}, 2*N, 2*N)
	T = tf/2
	if t<T
		aux[1: 2, :] .= intf(t, T, levels, α.*0, N)
		vector2matrix!(M, aux, σ)
		U0 = exp(-1im*M)
	else
		aux[1: 2, :] .= intf(T, T, levels, α.*0, N)
		vector2matrix!(M, aux, σ)
		U0 = exp(-1im*M)

		aux[1: 2, :] .= intf(t-T, T, levels, α, N)
		vector2matrix!(M, aux, σ)
		U0 = exp(-1im*M)*U0
	end
	return U0
end

function dU!(dU, U, args, t)
	H = args[1]
	N = args[2]::Int
	Ht = args[3]
	M = args[4]
	Ht .= H(t)
	for j = 1: N
		mul!(M, view(Ht,:,:,j), view(U,:,:,j))
		dU[:,:,j] .= (-1im).*M
	end
end

function solveU(H, χ, F, tf, N, σ)
    args = (σ, N)
	fH = @closure t-> H(t, F, args)

	U0 = zeros(Complex{Float64}, 2, 2, N)
	for j = 1: N
		U0[:, :, j] .= Matrix{Complex{Float64}}(I, 2, 2)
	end
	tspan = (zero(tf), tf)
	args2 = (fH, N, zeros(Complex{Float64}, 2, 2, N), zeros(Complex{Float64}, 2, 2))
	prob = DE.ODEProblem(dU!, U0, tspan, args2)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-14, maxiters=1e6)
	return sol(tf)
end

function U(system_params, σ; ideal_dynamics::Bool = false, matrix::Bool = false,
		   frame::String = "qubit")
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]
	N = system_params["N"]

	if frame == "qubit"
		Faux = @closure t-> F(t, tf, χ, levels, α, N, ideal_interaction=ideal_dynamics)
	elseif frame == "interaction"
		if ideal_dynamics == true
			Faux = @closure t-> zeros(4, N)
		else
			Faux = @closure t-> VI(t, tf, χ, levels, α, N)
		end
	elseif frame == "lab"
		Faux = @closure t-> Flab(t, tf, χ, levels, α, N; ideal_interaction=ideal_dynamics)
	end

	sol = solveU(H, χ, Faux, tf, N, σ)
	if matrix == false
		return sol
	else
		return tensor2matrix(sol)
	end
end

function solveOmega_snap(H, tf, order, N)
	numdims = 4*N
	fH = @closure t-> -reshape(H(t), (numdims))

	Omegaf = solveOmega(tf, order, fH, comm!, numdims)

	reshaped_Omegaf = zeros(order, 4, N)
	for i = 1: order
		reshaped_Omegaf[i, :, :] .= reshape(Omegaf[:, i], (4, N))
	end
	return reshaped_Omegaf

end
