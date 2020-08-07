import HomotopyContinuation;	const HC = HomotopyContinuation
import DynamicPolynomials;		const DP = DynamicPolynomials
using FastClosures
"""
	W0(t, tf, χ, args)
Calculate the correction operator, on the n subspace, associated to the a pulse
on the j subspace.
#Arguments
- t: time.
- tf: duration of the snap gate.
- χ: dispersive coupling constant
- n: subspace on which we calculate the correction operator.
- args: tuple with specification about the control
args[1] is an int that tells to which Pauli matrix the control couples to.
args[2] is the Fourier order of envelope.
args[3] is the envelope (fs or fc)
"""
function W0(t, tf, χ, args)
	control = args[1]
	order = args[2]
	envelope = args[3]

	W0t = zeros(4)
	if control == 1
		W0t[1] = envelope(t, tf, order)
	elseif control == 2
		W0t[2] = -envelope(t, tf, order)
	else
		W0t[3] = envelope(t, tf, order)/2
	end
	return W0t
end
"""
Calculate the correction operator W0 in the interaction picture.
#Arguments
- t: time.
- tf: duration of the snap gate.
- χ: dispersive coupling constant
- levels: the energy levels of the cavity that are resonantly driven by the ideal
Hamiltonian H0.
- α: the angle of the driving pulse in the second half of the time evolution.
- n: subspace on which we calculate the correction operator.
- W0_args: array with specification about the control; cf. W0.
- N: the size of the Hilbert space of the cavity.
"""
function W0I(t, tf, χ, levels, α, n, W0_args, N)
	W0t = W0(t, tf, χ, W0_args)
	indice = findall(k->k==n, levels)
	if length(indice) == 0
		W0It = W0t
	else
		T = tf/2.0
		if t < T + 10.0*eps()
			intfx, intfy = intf(t, T, zero(t))
			vn = (intfx, intfy, zero(t), zero(t))
			unitary_tranformation!(W0t, vn)
		else
			αn = α[indice[1]]
			intfx, intfy = intf(t - T, T, αn)
			vn = (intfx, intfy, zero(t), zero(t))
			unitary_tranformation!(W0t, vn)
			intfx, intfy = intf(T, T, zero(t))
			vn = (intfx, intfy, zero(t), zero(t))
			unitary_tranformation!(W0t, vn)
		end
	end
	return W0t
end
function delta_coeffs(system_params; eps=0.0)
	N = system_params["N"]
	M = zeros(3*N, 1)
	control = 3
	order = 0
	W0_args = (control, order, (t, tf, order)-> 1.0)
	indice = 1
	V(t) = zeros(4, N)
	for n=0:N-1
		Mn = view(M, n*3 + 1: (n + 1)*3)#zeros(3)
		linear_coeffs_col!(Mn, n, W0_args, indice, V, system_params)
		#M[n*3+1:(n+1)*3] .= Mn
	end

	#set elements smaller than eps to zero
	M[abs.(M) .< eps] .= 0.0
	return M
end
"""
Calculate the matrix with the coefficients of the free parameters in the
subspace n of the bosonic mode. The product of the coefficients with the
respective free parameters is the 1st order term of the Magnus expansion of the
correction operator. Each column is associated to one free parameter.
#Arguments
- n: the subspace of the bosonic mode.
- max_order: the maximum value of the parameter 'order'.
- V: The Hamiltonian in the interaction picture. It includes lower order
correction operators.
- system_params: dictionary with the system parameters.
#Keyword arguments
- eps: if the absolute value of any coefficients is smaller than eps, it will
be set to zero. The default value is 0.
"""
function linear_coeffs(n, max_order, V, system_params; eps::Real=0.0)
	M = zeros(3, 4*max_order)
	indice = 1
	for control = 1: 2
		for order = 1: max_order
			W0_args = (control, order, fc)
			linear_coeffs_col!(M, n, W0_args, indice, V, system_params)
			W0_args = (control, order, fs)
			linear_coeffs_col!(M, n, W0_args, indice + 1, V, system_params)
			indice += 2
		end
	end
	#set elements smaller than eps to zero
	M[abs.(M) .< eps] .= 0.0
	return M
end
"""
Update the elements of the matrix with the coefficients of the free parameters.
#Arguments
- M: the matrix with the coefficients of the free parameters.
- n: the subspace of the bosonic mode.
- W0_args: array with specifications about the control; cf. W0.
- indice: indice of the column being updated.
- V: The Hamiltonian in the interaction picture. It includes lower order
correction operators.
- system_params: dictionary with the system parameters.
"""
function linear_coeffs_col!(M, n, W0_args, indice, V, system_params)
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]
	N = system_params["N"]

	funcV = @closure t-> V(t)[:, n + 1]
	funcW = @closure t-> W0I(t, tf, χ, levels, α, n, W0_args, N)

	f0 = zeros(4, 3)
	tspan = (zero(tf), tf)
	args = (funcV, funcW, zeros(4), zeros(4))
	prob = DE.ODEProblem(df!, f0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-12)
	sol = sol(tf)

	M[:, indice] .= -(sol[1: 3, 2] .+ sol[1: 3, 3])
end
"""
Calculate the tensor with the coefficients of the quadratic products of the free
parameters in the subspace n of the bosonic mode. The coefficients (times the
quadratic products of the free parameters) appear in the 2nd term of the Magnus
expansion of the modified Hamiltonian (original Hamiltonian + correction
Hamiltonian).
#Arguments
- n: the subspace of the bosonic mode.
- max_order: the maximum value of the parameter 'order'.
- system_params: dictionary with the system parameters.
#Keyword arguments
- eps: if the absolute value of any coefficients is smaller than eps, it will
be set to zero. The default value is 0.
"""
function nonlinear_coeffs(n, max_order, system_params; eps::Real = 0.0)
	M = zeros(3, 4*max_order, 4*max_order)
	indice = 1
	for control = 1: 2
		for order = 1: max_order
			W0_args1 = (control, order, fc)
			nonlinear_coeffs_row!(M, n, max_order, W0_args1, indice, system_params)
			W0_args1 = (control, order, fs)
			nonlinear_coeffs_row!(M, n, max_order, W0_args1, indice+1, system_params)
			indice += 2
		end
	end
	#set elements smaller than eps to zero
	M[abs.(M) .< eps] .= 0.0
	return M
end
"""
Update the elements of the tensor with the coefficients of the quadratic products
of the free parameters.
#Arguments
- M: the tensor to be updated.
- n: the subspace of the bosonic mode.
- max_order: the maximum value of the parameter 'order'.
- W0_args1: array with specifications about control 1; cf. W0.
- indice1: indice of the elements being updated (2nd indice of the tensor).
- system_params: dictionary with the system parameters.
"""
function nonlinear_coeffs_row!(M, n, max_order, W0_args1, indice1, system_params)
	indice2 = 1
	for control = 1: 2
		for order = 1: max_order
			W0_args2 = (control, order, fc)
			index = (indice1, indice2)
			nonlinear_coeffs_col!(M, n, W0_args1, W0_args2, index, system_params)
			W0_args2 = (control, order, fs)
			index = (indice1, indice2 + 1)
			nonlinear_coeffs_col!(M, n, W0_args1, W0_args2, index, system_params)
			indice2 += 2
		end
	end
end
"""
Update the elements of the tensor with the coefficients of the quadratic products
of the free parameters.
#Arguments
- M: the tensor to be updated.
- n: the subspace of the bosonic mode.
- max_order: the maximum value of the parameter 'order'.
- W0_args1 and W0_args2: array with specifications about controls 1 and 2; cf. W0.
- index: index of the elements being updated (2nd and 3rd index of the tensor).
- system_params: dictionary with the system parameters.
"""
function nonlinear_coeffs_col!(M, n, W0_args1, W0_args2, index, system_params)
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]
	N = system_params["N"]

	funcW1 = @closure t-> W0I(t, tf, χ, levels, α, n, W0_args1, N)
	funcW2 = @closure t-> W0I(t, tf, χ, levels, α, n, W0_args2, N)

	f0 = zeros(4, 3)
	tspan = (zero(tf), tf)
	args = (funcW1, funcW2, zeros(4))
	prob = DE.ODEProblem(df2!, f0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-12)
	sol = sol(tf)

	M[:, index[1], index[2]] .= -sol[1: 3, 3]
end
"""
	df!(dOmega, Omega, args, t)
The integral of this function is related to the coefficients of the linear terms in the
polynomial system of equations.
"""
function df!(dOmega, Omega, args, t)
	V = args[1]
	W0 = args[2]
	temp1 = args[3]
	temp2 = args[4]
	#k=1
	dOmega[:, 1] .= -V(t)
	dOmega[:, 2] .= -W0(t)
	#k=2
	comm!(temp1, Omega[:, 2], dOmega[:, 1])
	comm!(temp2, Omega[:, 1], dOmega[:, 2])
	dOmega[:, 3] .= (temp1 .+ temp2)./2
	return nothing
end
"""
	df2!(dOmega, Omega, args, t)
The integral of this function is related to the coefficients of the nonlinear terms in the
polynomial system of equations.
"""
function df2!(dOmega, Omega, args, t)
	W01 = args[1]
	W02 = args[2]
	temp1 = args[3]::Array{Float64,1}
	#k=1
	dOmega[:,1] .= -W01(t)
	dOmega[:,2] .= -W02(t)
	#k=2
	comm!(temp1, Omega[:,2], dOmega[:,1])
	dOmega[:,3] .= temp1./2
	return nothing
end
"""
	correction_coeffs(Mlinear, Mnonlinear, Omega, max_order)
Generate the system of polynomial equations in a given number subspace and find its
solutions using homotopy continuation. Lagrange multipliers are used to find the solutions
with smallest norm.
"""
function correction_coeffs(Mlinear, Mnonlinear, Omega, max_order)
	HC.@polyvar x[1: 4*max_order]

	p1 = sum(Mlinear[1, :].*x) + sum(x.*(Mnonlinear[1, :, :]*x)) - Omega[1]
	p2 = sum(Mlinear[2, :].*x) + sum(x.*(Mnonlinear[2, :, :]*x)) - Omega[2]
	p3 = sum(Mlinear[3, :].*x) + sum(x.*(Mnonlinear[3, :, :]*x)) - Omega[3]
	norm2 = sum(x.*x)
	# Jacobians
	Jp1 = HC.differentiate(p1, x[1: end])
	Jp2 = HC.differentiate(p2, x[1: end])
	Jp3 = HC.differentiate(p3, x[1: end])
	Jnorm = HC.differentiate(norm2, x[1: end])
	# define Langrange equations
	HC.@polyvar l[1: 3]
	a = Jnorm - l[1]*Jp1 - l[2]*Jp2 - l[3]*Jp3
	C = vcat(a, [p1, p2, p3])

	res = HC.solve(C, show_progress=false)
	real_sols = HC.real_solutions(res)
	if length(real_sols) == 0
		res = HC.solve(C, show_progress=true)
		real_sols = HC.real_solutions(res)
	end

	real_sols = map(p -> p[1: 4*max_order], real_sols)

	coeffs = real_sols[1]
	for i = 2: length(real_sols)
		if norm(coeffs) > norm(real_sols[i])
			coeffs = real_sols[i]
		end
	end
	return coeffs
end
"""
	W(t, tf, χ, coeffs, max_order, Ncorr, N)
Correction Hamiltonian in the qubit frame.
# Arguments
- `t::Real`: time.
- `tf::Real`: gate time,
- `χ::Tuple`: Tuple with first, second, and third order dispersive couplings.
- `coeffs::Array`: corection coefficients.
- `max_order::Integer`: maximum number of harmonics of the correction pulse.
- `Ncorr::Integer`: bosonic subspace that should be corrected.
- `N::Integer`: number of dimensions of the truncated bosonic space.
"""
function W(t, tf, χ, coeffs, max_order, Ncorr, N)
	Wt = zeros(4, N)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	for j = 0: Ncorr - 1
		coeffsj = view(coeffs, 4*max_order*j + 1:4*max_order*(j+1))
		Wj!(Wt, t, tf, χef, j, coeffsj, max_order, N)
	end
	Wt[3, :] .= coeffs[end]/2
	return Wt
end
"""
	Wj!(Wt, t, tf, χef, j, coeffsj, max_order, N)
Update `Wt` in the bosonic subspace associated with the `j`th number state.
"""
function Wj!(Wt, t, tf, χef, j, coeffsj, max_order, N)
	gx, gy = g(t, tf, coeffsj, max_order)
	for n = 0: N - 1
		dφn = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
		dφj = χef[1]*j + χef[2]*j*(j - 1) + χef[3]*j*(j - 1)*(j - 2)
		φ = (dφn - dφj)*t
		cosφ = cos(φ)
		sinφ = sin(φ)
		Wt[1, n + 1] += + gx*cosφ - gy*sinφ
		Wt[2, n + 1] += - gx*sinφ - gy*cosφ
	end
	return nothing
end
"""
	g(t, tf, coeffs, max_order)
Correction pulse envelope.
# Arguments
- `t::Real`: time.
- `tf::Real`: gate time,
- `coeffs::Array`: corection coefficients.
- `max_order::Integer`: maximum number of harmonics of the correction pulse.
"""
function g(t, tf, coeffs, max_order)
	gx, gy = zero(t), zero(t)
	for order = 1: max_order
		cos_order = fc(t, tf, order)
		sin_order = fs(t, tf, order)

		l = 2*(order - 1) + 1
		gx += coeffs[l]*cos_order + coeffs[l + 1]*sin_order
		l += 2*max_order
		gy += coeffs[l]*cos_order + coeffs[l + 1]*sin_order
	end
	return gx, gy
end
"""
	WI(t, tf, χ, coeffs, max_order, levels, α, Ncorr, N)
Correction Hamiltonian in the interaction picture.
# Arguments
- `t::Real`: time.
- `tf::Real`: gate time,
- `χ::Tuple`: Tuple with first, second, and third order dispersive couplings.
- `coeffs::Array`: corection coefficients.
- `max_order::Integer`: maximum number of harmonics of the correction pulse.
- `levels::StaticArray`: levels in which one wants to imprint a phase.
- `α::Tuple`: the ''drinving angles'', which corresponds to π .- φ, where φ is the array with
angles that one wishes to imprint.
- `Ncorr::Integer`: bosonic subspace that should be corrected.
- `N::Integer`: number of dimensions of the truncated bosonic space.
"""
function WI(t, tf, χ, coeffs, max_order, levels, α, Ncorr, N)
	T = tf/2
	Wt = W(t, tf, χ, coeffs, max_order, Ncorr, N)
	v = zeros(4, N)
	if t < T + 10.0*eps()
		v[1: 2, :] .= intf(t, T, levels, α.*0, N)
		unitary_tranformation!(Wt, v, N)
	else
		v[1: 2, :] .= intf(t - T, T, levels, α, N)
		unitary_tranformation!(Wt, v, N)
		v[1: 2, :] .= intf(T, T, levels, α.*0, N)
		unitary_tranformation!(Wt, v, N)
	end
	return Wt
end
"""
	correction_coeffs2(Mlinear, Omega, max_order)
Calculate the correction coefficients using the fully linear method.
"""
function correction_coeffs2(Mlinear, Omega, max_order)
	if max_order > 1
		println("Danger!!! Correct correction_coeffs()")
	end
	coeffs = pinv(Mlinear)*Omega[1: 3]
	return coeffs
end
"""
	correction_matrix(max_order, system_params)
Correction matrix for the (linear) harmonic Magnus method.
"""
function correction_matrix(max_order, system_params)
	N = system_params["N"]
	χ = system_params["chi"]
	tf = system_params["tf"]
	levels = system_params["levels"]
	α = system_params["alpha"]
	Ncorr = system_params["Ncorr"]

	M = zeros(3*N, 4*max_order*N + 1)
	for i = 1: 4*max_order*N + 1
		coeffs = zeros(4*max_order*N + 1)
		coeffs[i] = 1.0
		func = @closure t-> WI(t, tf, χ, coeffs, max_order, levels, α, Ncorr, N)

		f0 = zeros(4, N)
		tspan = (zero(tf), tf)
		args = (func,)
		prob = DE.ODEProblem(df3!, f0, tspan, args)
		sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-10, abstol=1e-12)
		sol = sol(tf)

		M[:, i] = reshape(sol[1: 3, :], 3*N)
	end
	return M
end
"""
	df3!(dOmega, Omega, args, t)
Auxiliary function to compute the integral of a function W0(t).
"""
function df3!(dOmega, Omega, args, t)
	W0 = args[1]
	dOmega .= W0(t)
	return nothing
end
"""
	U(system_params, σ, coeffs, max_order; frame::String = "qubit", matrix::Bool = false)
Calculate the evolution operator.
# Keyword arguments
- `frame::String`: The frame in which the evolution must be calculated. The frame can be the
`qubit` frame, the `lab` frame, or the `interaction` frame. Default value is `qubit`.
- `matrix::Bool`: if `true`, U is returned as a matrix. If `false`, U is returned as a
tensor. Default value is `false`.
"""
function U(system_params, σ, coeffs, max_order; frame::String = "qubit",
		   matrix::Bool = false)
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]
	Ncorr = system_params["Ncorr"]
	N = system_params["N"]

	if frame == "qubit"
		Faux = @closure t-> (F(t, tf, χ, levels, α, N)
							 .+ W(t, tf, χ, coeffs, max_order, Ncorr, N))
	elseif frame == "interaction"
		Faux = @closure t-> (VI(t, tf, χ, levels, α, N)
							 .+ WI(t, tf, χ, coeffs, max_order, levels, α, Ncorr, N))
	elseif frame == "lab"
		Faux = @closure t-> Flab_corrected(t, tf, χ, levels, α, coeffs, max_order, Ncorr, N)
	end

	sol = solveU(H, χ, Faux, tf, N, σ)
	if matrix == false
		return sol
	else
		return tensor2matrix(sol)
	end
end
"""
	get_coeffs(magnus_order, max_order, Ncorr, system_params)
Calculate the correction coefficients.
"""
function get_coeffs(magnus_order, max_order, Ncorr, system_params)
	N = system_params["N"]
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]

	Faux = @closure t-> VI(t, tf, χ, levels, α, N)
	coeffs = get_coeffs_aux(Faux, magnus_order, max_order, system_params)
	return coeffs
end
"""
	get_coeffs(coeffs, magnus_order, max_order, Ncorr, system_params)
Calculate the correction coefficients.
"""
function get_coeffs(coeffs, magnus_order, max_order, Ncorr, system_params)
	N = system_params["N"]
	tf = system_params["tf"]
	χ = system_params["chi"]
	levels = system_params["levels"]
	α = system_params["alpha"]

	Faux = @closure t-> (VI(t, tf, χ, levels, α, N)
						 .+ WI(t, tf, χ, coeffs, max_order, levels, α, Ncorr, N))

	coeffs = get_coeffs_aux(Faux, magnus_order, max_order, system_params)
	return coeffs
end
"""
	get_coeffs_aux(Faux, magnus_order, max_order, system_params)
Calculate the correction coefficients.
"""
function get_coeffs_aux(Faux, magnus_order, max_order, system_params)
	tf = system_params["tf"]
	Ncorr = system_params["Ncorr"]
	N = system_params["N"]
	Omega = solveOmega_snap(Faux, tf, 2*magnus_order, N)
	Omega_sum = sum(Omega, dims=1)[1, :, :]

	coeffs = zeros(N*4*max_order + 1)
	for n = 0: Ncorr - 1
		Omega_sum_n = Omega_sum[:, n + 1]
		Mlinear = linear_coeffs(n, max_order, Faux, system_params; eps=1e-10)
		Mnonlinear = nonlinear_coeffs(n, max_order, system_params; eps=1e-10)
		coeffsn = correction_coeffs(Mlinear, Mnonlinear, Omega_sum_n, max_order)

		index = 4*max_order*n + 1: 4*max_order*(n+1)
		coeffs[index] .= coeffsn
	end
	return coeffs
end
"""
	fcorrected(t, tf, α, coeffs)
Calculate corrected envelope function in number state subspace.
"""
function fcorrected(t, tf, α, coeffs)
	gx, gy = g(t, tf, coeffs, max_order)
	theta = pi/2
	T = tf/2
	if t < T
		fx = theta*fc(t, T, 1) + gx
		fy = gy
	else
		fct = theta*fc(t - T, T, 1)
		fx = fct*cos(α) + gx
		fy = fct*sin(α) + gy
	end
	return fx, fy
end
"""
	Fcorrected(t, tf, χ, levels, α, coeffs, Ncorr, N)
Calculate the corrected driving Hamiltonian in the qubit frame.
"""
function Fcorrected(t, tf, χ, levels, α, coeffs, Ncorr, N)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	Ft = zeros(4, N)
	for i = 1: length(levels)
		Fj!(Ft, t, tf, χef, levels[i], α[i], N, false)
	end
	for i = 1: Ncorr
		coeffsj = view(coeffs, 4*max_order*(i - 1) + 1: 4*max_order*i)
		Wj!(Ft, t, tf, χef, i - 1, coeffsj, max_order, N)
	end
	return Ft
end
"""
	correction_pulse_envelope(t, tf, χ, coeffs, max_order, Ncorr)
Calculate the correction pulse envelope in the lab frame.
"""
function correction_pulse_envelope(t, tf, χ, coeffs, max_order, Ncorr)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	gx_lab, gy_lab = zero(t), zero(t)
	for n = 0: Ncorr - 1
		coeffsn = view(coeffs, 4*max_order*n + 1: 4*max_order*(n+1))
		gx, gy = g(t, tf, coeffsn, max_order)
		dφn = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
		φ = dφn*t
		cosφ, sinφ = cos(φ), sin(φ)
		gx_lab += gx*cosφ + gy*sinφ
		gy_lab += gx*sinφ - gy*cosφ
	end
	return gx_lab, gy_lab
end
"""
	pulse_envelope(t, tf, χ, levels, α)
Calculate the (uncorrected) pulse envelope in the lab frame
"""
function pulse_envelope(t, tf, χ, levels, α)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	fx_lab, fy_lab = zero(t), zero(t)
	for i = 1: length(levels)
		fx, fy = f(t, tf, α[i])
		li = levels[i]
		dφ = χef[1]*li + χef[2]*li*(li - 1) + χef[3]*li*(li - 1)*(li - 2)
		φ = dφ*t
		cosφ, sinφ = cos(φ), sin(φ)
		fx_lab += fx*cosφ + fy*sinφ
		fy_lab += fx*sinφ - fy*cosφ
	end
	return fx_lab, fy_lab
end
"""
	Flab(t, tf, χ, levels, α, N; ideal_interaction::Bool=false)
Calculate the driving Hamiltonian in the lab frame.
"""
function Flab(t, tf, χ, levels, α, N; ideal_interaction::Bool=false)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	Ft = zeros(4, N)
	if ideal_interaction == true
		for i = 1: length(levels)
			indice = levels[i] + 1
			fx, fy = f(t, tf, α[i])
			li = levels[i]
			dφ = χef[1]*li + χef[2]*li*(li - 1) + χef[3]*li*(li - 1)*(li - 2)
			φ = dφ*t
			cosφ, sinφ = cos(φ), sin(φ)
			Ft[1, indice] = fx*cosφ + fy*sinφ
			Ft[2, indice] = fx*sinφ - fy*cosφ
		end
		for i = 1: N
			n = i - 1
			ω = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
			Ft[3, i] = ω/2
			#Ft[3, i] = χ*(i-1)/2.0
		end
	else
		fx, fy = pulse_envelope(t, tf, χ, levels, α)
		for i = 1: N
			Ft[1, i] = fx
			Ft[2, i] = fy
			n = i - 1
			ω = χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2)
			Ft[3, i] = ω/2
			#Ft[3, i] = χ*(i-1)/2.0
		end
	end
	return Ft
end
"""
	Flab_corrected(t, tf, χ, levels, α, coeffs, max_order, Ncorr, N)
Calculate the corrected driving Hamiltonian in the lab frame.
"""
function Flab_corrected(t, tf, χ, levels, α, coeffs, max_order, Ncorr, N)
	χef = (χ[1], χ[2]/2, χ[3]/6)
	Ft = zeros(4, N)
	original_pulse = pulse_envelope(t, tf, χ, levels, α)
	correction_pulse = correction_pulse_envelope(t, tf, χ, coeffs, max_order, Ncorr)
	fx, fy =  original_pulse .+ correction_pulse
	for i = 1: N
		Ft[1, i] = fx
		Ft[2, i] = fy
		n = i - 1
		Ft[3, i] = (χef[1]*n + χef[2]*n*(n - 1) + χef[3]*n*(n - 1)*(n - 2))/2
		#Ft[3, i] = χ*(i-1)/2.0
	end
	return Ft
end
