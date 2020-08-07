using DifferentialEquations
using FastClosures
"""
	comm!(commAB, A, B)
Calculate the commutator of the operators A and B, and save the result in commAB.
The operators are written in the basis { a^2 + a†^2, -i(a^2 - a†^2), 2a†a+1 }.
"""
function comm!(commAB, A, B)
	commAB[1] = 4*(A[3]*B[2] - A[2]*B[3])
	commAB[2] = 4*(A[1]*B[3] - A[3]*B[1])
	commAB[3] = 4*(A[1]*B[2] - A[2]*B[1])
	return nothing
end
"""
	Λ(t, tf)
Definite integral of the driving envelope.
"""
function Λ(t, tf)
	x = t/tf
	return (x - sin(2*pi*x)/(2*pi))
end
"""
	λ(t, tf)
Driving envelope.
"""
function λ(t, tf)
	x = t/tf
	return (1 - cos(2*pi*x))/tf
end
"""
	H(t, tf; cr_terms::Bool=true)
Hamiltonian of a parametrically driven cavity.
"""
function H(t, tf; cr_terms::Bool=true)
	λt = λ(t, tf)
	Ht = zeros(3)
	if cr_terms == true
		Ht[1] = λt*sin(4*t)/2
		Ht[2] = λt*(1 - cos(4*t))/2
		Ht[3] = λt*sin(2*t)
	else
		Ht[2] = λt/2
	end
	return Ht
end
"""
	V_I(t, tf)
Error Hamiltonian V in the interaction picture.
"""
function V_I(t, tf)
	λt = λ(t, tf)
	Λt = Λ(t, tf)

	V_It = zeros(3)
	V_It[1] = λt*(sin(2*t)*sinh(2*Λt) + sin(4*t)*cosh(2*Λt)/2)
	V_It[2] = -λt*cos(4*t)/2
	V_It[3] = λt*(sin(2*t)*cosh(2*Λt) + sin(4*t)*sinh(2*Λt)/2)
	return V_It
end
"""
	dv!(dv, v, args, t)
Heisenberg equations for v = [a, a†].
"""
function dv!(dv, v, args, t)
	H = args[1]
	tf = args[2]
	cr_terms = args[3]
	Ht = H(t)
	a = view(v, :, :, 1)
	adagger = view(v, :, :, 2)
	dv[:, :, 1] .= (-2im*Ht[1] + 2*Ht[2])*adagger .- 2im*Ht[3]*a
	dv[:, :, 2] .= (+2im*Ht[1] + 2*Ht[2])*a .+ 2im*Ht[3]*adagger
end
"""
	d2moments!(dv, v, args, t)
Differential equations for the second moments (v = [a^2 + a†^2, -i(a^2 - a†^2),
aa† + a†a]). Dissipation is taken in account. `H` is a function the returns the Hamiltonian
at time `t`.
"""
function d2moments!(dv, v, args, t)
	H = args[1]
	tf = args[2]
	κ = args[3]
	Ht = args[4]

	Ht .= H(t)
	freal = 2*Ht[2]
	fimag = -2*Ht[1]
	#greal = -κ/2.0
	gimag = -2*Ht[3]
	dv[1] = -κ*v[1] - 2*gimag*v[2] + 2*freal*v[3]
	dv[2] = 2*gimag*v[1] - κ*v[2] + 2*fimag*v[3]
	dv[3] = 2*freal*v[1] + 2*fimag*v[2] - κ*v[3] + κ
end
"""
	solve_2moments(tf, κ, cr_terms, coeffs)
Solve equations of motions for the second moments.
"""
function solve_2moments(tf, κ, cr_terms, coeffs)
	Hmod = @closure t-> H(t, tf; cr_terms=cr_terms) .+ WharmonicR(t, tf, coeffs)
	f0 = [0.0, 0.0, 1.0]
	tspan = (0.0, tf)
	args = (Hmod, tf, κ, zeros(3))
	prob = ODEProblem(d2moments!, f0, tspan, args)
	sol = solve(prob, Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-12)
	return sol(tf)
end
