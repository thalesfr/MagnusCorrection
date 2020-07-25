using StaticArrays
using FastClosures
"""
gellmann_matrices(;type=Float64)
Return the Gell-Mann matrices.
"""
function gellmann_matrices(;type=Float64, struct_const=false)
	λ1 = SMatrix{3, 3, Complex{type}}([0 1 0; 1 0 0; 0 0 0])
	λ2 = SMatrix{3, 3, Complex{type}}([0 -1im 0; 1im 0 0; 0 0 0])
	λ3 = SMatrix{3, 3, Complex{type}}([1 0 0; 0 -1 0; 0 0 0])
	λ4 = SMatrix{3, 3, Complex{type}}([0 0 1; 0 0 0; 1 0 0])
	λ5 = SMatrix{3, 3, Complex{type}}([0 0 -1im; 0 0 0;1im 0 0])
	λ6 = SMatrix{3, 3, Complex{type}}([0 0 0; 0 0 1; 0 1 0])
	λ7 = SMatrix{3, 3, Complex{type}}([0 0 0; 0 0 -1im; 0 1im 0])
	λ8 = SMatrix{3, 3, Complex{type}}(
		 [1/sqrt(type(3)) 0 0; 0 1/sqrt(type(3)) 0; 0 0 -2/sqrt(type(3))])
	λ9 = SMatrix{3, 3, Complex{type}}([1 0 0; 0 1 0; 0 0 1])
	λ = (λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8, λ9)

	if struct_const == true
		index = ((1, 2, 3), (1, 4, 7), (1, 6, 5), (2, 4, 6), (2, 5, 7), (3, 4, 5),
				 (3, 7, 6), (4, 5, 8), (6, 7, 8))

		f = zeros(type, 9)
		f[1] = type(1)
		f[2: 7] .= type(1)/2
		f[8: 9] .= sqrt(type(3))/2
		f = Tuple(f)

		return λ, index, f
	else
		return λ
	end
end

function vector2matrix(v, λ)
	M = v[1]*λ[1]
	for i = 2: 9
		@inbounds M = M + v[i]*λ[i]
	end
	return M
end
"""
	fidelity(U, U_ideal, dims)
Fidelity of the evolution operator U with respect to the ideal evolution operator U_ideal.
The Hilbert space is truncated in the `ndims` first dimensions.
"""
function fidelity(U, U_ideal, dims)
	U_ideal_dag = adjoint(U_ideal)
	M = U_ideal_dag[1: dims, 1: dims]*U[1: dims, 1: dims]
	M_dag = adjoint(M)
	fidelity = (tr(M*M_dag) + abs(tr(M))^2)/(dims*(dims + 1))
	return real(fidelity)
end
"""
	gell_mann_decomp(M)
Decomposition of a generic 3x3 Hermitian matrix M in the Gell-Mann basis.
"""
function gell_mann_decomp(M)
	c = zeros(9)
	c[9] = real(M[1,1] + M[2,2] + M[3,3])/3
	c[8] = (-real(M[3,3]) + c[9])*sqrt(3.0)/2
	c[1], c[2] = real(M[2,1]), imag(M[2,1])
	c[3] = real(M[1,1]) - c[9] - c[8]/sqrt(3.0)
	c[4], c[5] = real(M[3,1]), imag(M[3,1])
	c[6], c[7] = real(M[3,2]), imag(M[3,2])
	return c
end
"""
	H0(t, α, η, f; cr_terms::Bool = false)
Hamiltonian of an isolated 3 level system in the rotating frame.
# Arguments
- `t::Real`: time.
- `α::Real`: anarmonicity.
- `η::Real`: parameter that characterizes the strength of the 1<->2 transition relative
to the 0<->1 transition.
- `f::Function`: pulse envelope as a sunction of t.
- `cr_terms::Bool = false`: If true, counter rotating terms are taken in account.
"""
function H0(t, α, η, f; cr_terms::Bool=false)
	type = typeof(t)
	ft = f(t)/2
	if cr_terms == false
		H0t = @SVector [ft,
						0,
						0,
						ft*η,
						0,
						0,
						0,
						-α/sqrt(type(3)),
						α/3]
	else
		cos2t = cos(2*t)
		sin2t = sin(2*t)
		H0t = @SVector [ft*(1 + cos2t),
						-ft*sin2t,
						0,
						ft*(1 + cos2t)*η,
						ft*sin2t*η,
						0,
						0,
						-α/sqrt(type(3)),
						α/3]
	end
	return H0t
end
#Differential equation function of the evolution operator.
function dU!(dU, U, args, t)
	Omega = args[1]
	λ = args[2]
	H = args[3]
	Omega_t = args[4]

	Omega_t = Omega(t)
	#Hamiltonian
	H .= zero(Complex{typeof(t)})
	for i = 1: 9
		H .= H .+ Omega_t[i]*λ[i]
	end
	H .= -1im*H
	mul!(dU, H, U)
end
function dU(U, args, t)
	Ω = args[1]
	λ = args[2]

	Ωt = Ω(t)
	H = vector2matrix(Ωt, λ)
	dU = -1im*H*U
	return dU
end

function solveU2(H, tf, λ; type::DataType = Float64)
	U0 = Matrix{Complex{type}}(I, 3, 3)
	tspan = (type(0), tf)
	args = (H, λ, zeros(Complex{type}, 3, 3), zeros(9))
	prob = DE.ODEProblem(dU!, U0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-11, abstol=1e-14)
	if sol.retcode == :Success
		return sol
	else
		error("Could not obtain the evolution operator. Numerical integration was
			  unsuccessful.")
	end
end
function solveU(H, tf, λ; type::DataType = Float64)
	U0 = SMatrix{3,3}(Matrix{Complex{type}}(I, 3, 3))
	tspan = (zero(tf), tf)
	args = (H, λ)
	prob = DE.ODEProblem(dU, U0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-11, abstol=1e-14)
	if sol.retcode == :Success
		return sol
	else
		error("Could not obtain the evolution operator. Numerical integration was
			  unsuccessful.")
	end
end
"""
	comm_ijk!(commAB, A, B, i, j, k, f_ijk)
.
"""
function comm_ijk!(commAB, A, B, i, j, k, f_ijk)
	commAB[k] = commAB[k] + (A[i]*B[j] - A[j]*B[i])*2*f_ijk
	commAB[j] = commAB[j] + (A[k]*B[i] - A[i]*B[k])*2*f_ijk
	commAB[i] = commAB[i] + (A[j]*B[k] - A[k]*B[j])*2*f_ijk
	return nothing
end
"""
	comm!(commAB, A, B)
Compute the commutator of A and B. The result is stored in commAB.
"""
# function comm!(commAB, A, B)
# 	type = eltype(A)
# 	commAB .= zero(type)
# 	w1, w2 = type(1)/2, sqrt(type(3))/2
# 	comm_ijk!(commAB, A, B, 1, 2, 3, 1)
# 	comm_ijk!(commAB, A, B, 1, 4, 7, w1)
# 	comm_ijk!(commAB, A, B, 1, 6, 5, w1)
# 	comm_ijk!(commAB, A, B, 2, 4, 6, w1)
# 	comm_ijk!(commAB, A, B, 2, 5, 7, w1)
# 	comm_ijk!(commAB, A, B, 3, 4, 5, w1)
# 	comm_ijk!(commAB, A, B, 3, 7, 6, w1)
# 	comm_ijk!(commAB, A, B, 4, 5, 8, w2)
# 	comm_ijk!(commAB, A, B, 6, 7, 8, w2)
# 	return nothing
# end
const index_comm = ((1, 2, 3), (1, 4, 7), (1, 6, 5),
					(2, 4, 6), (2, 5, 7), (3, 4, 5),
					(3, 7, 6), (4, 5, 8), (6, 7, 8))

type = Float64
f = zeros(type, 9)
f[1] = 2*type(1)
f[2: 7] .= type(1)
f[8: 9] .= sqrt(type(3))
const struct_const = Tuple(f)
function comm!(commAB, A, B)
	commAB .= zero(eltype(A))
	for n = 1: 9
		i, j, k = index_comm[n]
		@inbounds commAB[k] += (A[i]*B[j] - A[j]*B[i])*struct_const[n]
		@inbounds commAB[j] += (A[k]*B[i] - A[i]*B[k])*struct_const[n]
		@inbounds commAB[i] += (A[j]*B[k] - A[k]*B[j])*struct_const[n]
	end
	return nothing
end
"""
	V_I(t, α, η, f, c; cr_terms::Bool = false)
Hamiltonian in the interaction picture with respect to H0.
# Arguments
- `t::Real`: time.
- `α::Real`: anarmonicity.
- `η::Real`: parameter that characterizes the strength of the 1<->2 transition relative
to the 0<->1 transition.
- `f::Function`: pulse envelope as a sunction of t.
- `c::Real`: integral of the pulse envelope as a function of t.
- `cr_terms::Bool = false`: If true, counter rotating terms are taken in account.
"""
function V_I(t, α, η, f, c; cr_terms::Bool = false)
	type = typeof(t)
	ft = f(t)/2		#note the division by 2!!!
	ct = c(t)
	cosc, sinc = cos(ct), sin(ct)
	cosα, sinα = cos(α*t), sin(α*t)
	ηf = η*ft

	if cr_terms == true
		cos_dp2, sin_dp2 = cos((α + 2)*t), sin((α + 2)*t)
		V_It = @SVector [ft*cos(2*t),
						 -ft*sin(2*t)*cos(2*ct),
						 ft*sin(2*t)*sin(2*ct),
						 ηf*cosc*(cosα + cos_dp2),
						 ηf*cosc*(sinα + sin_dp2),
						 ηf*sinc*(sinα + sin_dp2),
						 -ηf*sinc*(cosα + cos_dp2),
						 0,
						 0]
	else
		V_It = @SVector [0,
						 0,
						 0,
						 ηf*cosc*cosα,
						 ηf*cosc*sinα,
						 ηf*sinc*sinα,
						 -ηf*sinc*cosα,
						 0,
						 0]
	end
	return V_It
end
"""
	U(system_params, λ; ideal_dynamics=false)
Evolution operator.
# Arguments
- `system_params::Dictionary`: dictionary with the parameters of the system.
- `λ::Array{Array{<:Complex, 2}}`: Array with the Gell-Mann
"""
function U(system_params, λ; ideal_dynamics::Bool = false)
	tf = system_params["tf"]
	α = system_params["alpha"]
	η = system_params["eta"]
	pulse = system_params["pulse"]

	if ideal_dynamics == true
		η = zero(η)
	end
	H = @closure t-> H0(t, α, η, pulse)
	sol = solveU2(H, tf, λ)
	return sol(tf)
end
#Drag correction of the leakage.
function correction_drag(t, α, η, pulse, pulse_derivative; high_order::Bool = false)
	f = pulse(t)
	f2 = f*f
	df = pulse_derivative(t)
	η2 = η*η

	if high_order == true
		η4 = η2*η2
		f3, f4 = f2*f, f2*f2
		f5 = f4*f
		Δ = (η2 - 4)*f2/(4*α) - (η4 - 7*η2 + 12)*f4/(16*α^3)
		gx = (η2 - 4)*f3/(8*α^2) - (13*η4 - 76*η2 + 112)*f5/(128*α^4)
		gy = -df/α + 33*(η2 - 2)*f2*df/(24*α^3)
	else
		Δ = (η2 - 4)*f2/(4*α)
		gx = zero(t)
		gy = -df/α
	end
	return Δ, gx, gy
end
#Correction Hamiltonian for drag.
function Wdrag(t, α, η, pulse, pulse_derivative; high_order::Bool = false)
	type = typeof(t)
	Δt, gx, gy = correction_drag(t, α, η, pulse, pulse_derivative; high_order=high_order)
	cost, sint = cos(t), sin(t)
	gt = gx*cost + gy*sint
	ηg = η*gt
	Δprime = Δt/2

	Wt = @SVector [gt*cost,
				   -gt*sint,
				   Δprime,
				   ηg*cost,
				   ηg*sint,
				   0,
				   0,
				   -(sqrt(type(3))*Δprime),
				   Δprime]
	return Wt
end

function Udrag(system_params, λ; high_order::Bool = false)
	tf = system_params["tf"]
	α = system_params["alpha"]
	η = system_params["eta"]
	pulse = system_params["pulse"]
	pulse_derivative = system_params["pulse_derivative"]

	H_drag = @closure (
		t-> H0(t, α, η, pulse)
		+ Wdrag(t, α, η, pulse, pulse_derivative, high_order= high_order))

	sol = solveU(H_drag, tf, λ)
	return sol(tf)
end
#Lindbladian
function L(ρ, c)
	c_dagger = adjoint(c)
	n = c_dagger*c
	return c*ρ*c_dagger - (n*ρ + ρ*n)/2
end

function dρ(ρ, args, t)
	H = args[1]
	c1 = args[2]
	c2 = args[3]
	λ = args[4]

	Ht = vector2matrix(H(t), λ)
	return 1im*(ρ*Ht - Ht*ρ) + L(ρ, c1) + L(ρ, c2)
end

function solveρ(system_params, λ, c1, c2, ρ0; type::DataType = Float64)
	tf = system_params["tf"]
	α = system_params["alpha"]
	η = system_params["eta"]
	pulse = system_params["pulse"]

	H = @closure t-> H0(t, α, η, pulse)

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

function ρbloch(a)
	u, v, w = a
	ρ11, ρ12 = (1 + w)/2, (u - 1im*v)/2
	ρ21, ρ22 = (u + 1im*v)/2, (1 - w)/2
	ρ = @SMatrix [ρ11 ρ12 0; ρ21 ρ22 0; 0 0 0]
	return ρ
end

function average_fidelity(system_params, λ, c1, c2)
	tf = system_params["tf"]
	Uideal = U(system_params, λ; ideal_dynamics = true)

	dict = Dict("+x"=> (1.0, 0.0, 0.0), "-x"=> (-1.0, 0.0, 0.0),
				"+y"=> (0.0, 1.0, 0.0), "-y"=> (0.0, -1.0, 0.0),
				"+z"=> (0.0, 0.0, 1.0), "-z"=> (0.0, 0.0, -1.0))
	F = 0.0
	for tag in ("+x", "-x", "+y", "-y", "+z", "-z")
		a = dict[tag]
		ρ0 = ρbloch(a)
		ρf = solveρ(system_params, λ, c1, c2, ρ0)(tf)
		ρfideal = Uideal*ρ0*adjoint(Uideal)
		F += tr(ρfideal*ρf)/6
	end
	if imag(F) > 10.0^(-10.0)
		error("Complex fidelity!")
	else
		return real(F)
	end
end
