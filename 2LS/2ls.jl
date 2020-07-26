using DifferentialEquations
using StaticArrays
using FastClosures
#the Pauli matrices
function pauli_matrices(; type=Float64)
	σ1 = SMatrix{2, 2, Complex{type}}([0 1; 1 0])
	σ2 = SMatrix{2, 2, Complex{type}}([0 -1im; 1im 0])
	σ3 = SMatrix{2, 2, Complex{type}}([1 0; 0 -1])
	σ4 = SMatrix{2, 2, Complex{type}}([1 0; 0 1])
	σ = (σ1, σ2, σ3, σ4)
	return σ
end

function fidelity(U, U_ideal, dims)
	U_ideal_dag = adjoint(U_ideal)
	M = U_ideal_dag[1: dims, 1: dims]*U[1: dims, 1: dims]
	M_dag = adjoint(M)
	fidelity = (tr(M*M_dag) + abs(tr(M))^2)/(dims*(dims + 1))
	return real(fidelity)
end
#Decompose a generic 2x2 Hermitian matrix M in the Pauli basis.
function pauli_decomp(M)
	c = zeros(4)
	c[1] = real(M[1,2])
	c[2] = imag(M[2,1])
	c[3] = real(M[1,1] - M[2,2])/2
	c[4] = real(M[1,1] + M[2,2])/2
	return c
end
#Definite integral of the envelope function f.
function c(t, tf)
	theta = pi/2
	x = t/tf
	c_t = (x - sin(2*pi*x)/(2*pi))/2
	return theta*c_t
end
#Envelope function.
function f(t, tf; derivative::Bool = false)
	theta = pi/2
	x = t/tf
	if derivative == false
		return theta*(1 - cos(2*pi*x))/tf
	else
		return theta*sin(2*pi*x)*2*pi/(tf*tf)
	end
end
#Hamiltonian of an isolated 2 level system in the rotating frame.
function H0(t, tf, cr_terms)
	H0t = zeros(4)
	ft = f(t, tf)/2
	if cr_terms == false
		H0t[1] = ft
	else
		H0t[1] = ft*(1 + cos(2*t))
		H0t[2] = -ft*sin(2*t)
	end
	return H0t
end
#Differential equation function of the evolution operator.
function dU!(dU, U, args, t)
	Omega = args[1]
	σ = args[2]
	H = args[3]
	Omega_t = args[4]

	Omega_t = Omega(t)
	#Hamiltonian
	H .= zero(t)
	for i = 1: 4
		H .= H .+ Omega_t[i]*σ[i]
	end
	H .= -1im*H
	mul!(dU,H,U)
end
#Commutator of operators A and B.
function comm!(commAB, A, B)
	commAB[1] = (A[2]*B[3] - A[3]*B[2])*2
	commAB[2] = (A[3]*B[1] - A[1]*B[3])*2
	commAB[3] = (A[1]*B[2] - A[2]*B[1])*2
	return nothing
end
#Hamiltonian of a 2 level system in the interaction picture.
function V_I(t, tf)
	ft = f(t, tf)/2		#note the division by 2!!!
	c_t = c(t, tf)

	V_It = zeros(4)
	V_It[1] = ft*cos(2*t)
	V_It[2] = -ft*sin(2*t)*cos(2*c_t)
	V_It[3] = ft*sin(2*t)*sin(2*c_t)
	return V_It
end
#The lindbladian superoperator.
#ρ is the density operator. c is the collapse opperator.
function lindbladian!(lρ, c, ρ)
	cdag = adjoint(c)
	m = cdag*c
	lρ .= ρ*cdag
	lρ .= c*lρ .- (m*ρ .+ ρ*m)./2
	return nothing
end
#The master equations for the two level system.
#H is the Hamiltonian. c_list is a tuple with the collapse operators.
function dρ!(dρ, ρ, args, t)
	H = args[1]
	clist = args[2]
	ncollapse = arg[3]
	Ht = args[3]
	Hmatrix = args[4]
	lρ = args[5]

	Ht .= H(t)
	Hmatrix .= Ht[1].*σ[1]
	for i = 2: 4
		Hmatrix .= Hmatrix .+ Ht[i].*σ[i]
	end
	#unitary evolution
	dρ .= 1im.*(ρ*Hmatrix .- Hmatrix*ρ)
	#non-unitary evolution
	for i = 1: ncollapse
		lindbladian!(lρ, c1, ρ)
		dρ .= dρ .+ lρ
	end
end
#
function solveU(H, tf, σ)
	U0 = Matrix{Complex{Float64}}(I, 2, 2)
	tspan = (zero(tf), tf)
	args = (H, σ, zeros(Complex, 2, 2), zeros(4))
	prob = ODEProblem(dU!, U0, tspan, args)
	sol = solve(prob, Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-14)
	return sol
end
#
function U(tf, σ; ideal_dynamics=false)
	cr_terms = !ideal_dynamics
	H = @closure t-> H0(t, tf, cr_terms)
	sol = solveU(H, tf, σ)
	return sol(tf)
end
#
function solveρ(tf, H, clist, ρ0)
	ρ0 = convert(Array{Complex{Float64}}, ρ0)
	#clist = (sqrt(γ1)*[[0 0];[1.0 0]], sqrt(γ2)*σ3)
	tspan = (zero(tf), tf)
	args = (H, clist, length(clist), zeros(4), zeros(Complex{Float64}, 2, 2),
			zeros(Complex{Float64}, 2, 2), zeros(Complex{Float64}, 2, 2))
	prob = ODEProblem(dρ!, ρ0, tspan, args)
	sol = solve(prob, Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-14)
	return sol(tf)
end
#This function has the Bloch vector a as input and returns the density operator.
function ρbloch(a)
	u, v, w = a[1], a[2], a[3]
	ρ11, ρ12 = 1 + w, u - 1im*v
	ρ21, ρ22 = u + 1im*v, 1 - w
	#ρ = [[1+w u-1im*v];[u+1im*v 1-w]]/2
	ρ = [ρ11 ρ12; ρ21 ρ22]/2
	return ρ
end
#This function calculates the average fidelity in a two level system.
#Hideal is the ideal Hamiltonian; H is the nonideal Hamiltonian; γ1 and γ2 are
#the dissipation and dephasing constants.
function average_fidelity(tf, H, Hideal, clist)
	U = solveU(Hideal, tf, σ)
	dict = Dict("+x"=> (1,0,0), "-x"=> (-1,0,0),
				"+y"=> (0,1,0), "-y"=> (0,-1,0),
				"+z"=> (0,0,1), "-z"=> (0,0,-1))

	F = zero(tf)
	for tag in ("+x", "-x", "+y", "-y", "+z", "-z")
		a = dict[tag]
		ρ0 = ρbloch(a)
		ρf = solveρ(tf, H, clist, ρ0)
		ρfideal = U*ρ0*adjoint(U)
		F += tr(ρfideal*ρf)/6
	end
	if imag(F) > 10.0^(-10.0)
		error("Complex fidelity!")
		return -1
	else
		return real(F)
	end
end
