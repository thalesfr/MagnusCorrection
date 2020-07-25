
import DifferentialEquations;	const DE = DifferentialEquations
"""
	dOmega!(dOmega, Omega, args, t)
Differential equation for the first terms of the Magnus expansion.
#Arguments
- dOmega: 2-dimensional array with the time derivative of the Magnus expansion
terms.
- Omega: 2-dimensional array with the Magnus expansion terms. The nth column is
the nth term of the Magnus expansion: Omega[:,n] = Omega_n.
- args: Array with the many different arguments of the function.
args[1] = order: The number of terms of the Magnus expansion that will be computed.\n
args[2] = A: the matrix of function that defines the linear ODE (dx/dt = A*x).\n
args[3] = S: array of arrays used to store the S operators. ``S[n] = S_n``, where
S[n] is a 2-dimensional array. ``(S[n])[:,j] = S_n^{(j)}``.\n
args[4] = B: array where B[n] = b_n/n!. b_n is the n'th Bernoulli number.\n
args[5] = Snj: array to temporarily store S_n^{(j)}\n
args[6] = temp1 and args[7] = temp2: temporary arrays to store data.\n
- t: time variable.\n
"""
function dOmega!(dOmega, Omega, args, t)
	order = args[1]::Integer
	A = args[2]
	comm! = args[3]
	S = args[4]
	B = args[5]
	Snj = args[6]
	temp = args[7]
	temp2 = args[8]

	dOmega[:, 1] .= A(t)

	S2 = S[2]
	Omega1 = view(Omega, :, 1)
	dOmega1 = view(dOmega, :, 1)
	comm!(Snj, Omega1, dOmega1)
	S2[:, 1] .= Snj
	dOmega[:, 2] .= B[1].*Snj
	for n = 3: order
		Sn = S[n]
		update_S!(Snj, S, n, 1, comm!, Omega, Omega1, dOmega1, temp2)
		temp .= B[1].*Snj
		for j = 2: n - 1
			update_S!(Snj, S, n, j, comm!, Omega, Omega1, dOmega1, temp2)
			if j%2 == 0
				temp .= temp .+ B[j].*Snj
			end
		end
		dOmega[:, n] .= temp
	end
	return nothing
end
"""
	update_S!(Snj, S, n, j, Omega, Omega1, dOmega1, temp2)
Calculate ``S_n^{(j)}`` recursivelly and store the result in S and Snj. The result
is stored in Snj just for efficiency purposes.
#Arguments
See dOmega! documentation.
"""
function update_S!(Snj, S, n, j, comm!, Omega, Omega1, dOmega1, temp2)
	Sn = S[n]
	if j == 1
		Omegam = view(Omega, :, n - 1)
		comm!(Snj, Omegam, dOmega1)
	else
		comm!(Snj, Omega1, view(S[n - 1], :, j - 1))
		for m = 2: n - j
			Omegam = view(Omega, :, m)
			comm!(temp2, Omegam, view(S[n - m], :, j - 1))
			Snj .= Snj .+ temp2
		end
	end
	Sn[:, j] .= Snj
	return nothing
end
"""
	bernoulli(n)
Calculate the n'th Bernoulli number. This function was taken from Rosetta Code
(https://rosettacode.org/wiki/Bernoulli_numbers#Julia).
#Arguments:
- n: the index of the Bernoulli number.
"""
function bernoulli(n)
    A = Vector{Rational{Int64}}(undef, n + 1)
    for m = 0: n
        A[m + 1] = 1//(m + 1)
        for j = m: -1: 1
            A[j] = j*(A[j] - A[j + 1])
        end
    end
    return A[1]
end
"""
	solveOmega(tf, order, A, numdims)
Calculate the Magnus expansion.
#Arguments
- tf: the time at which the Magnus expansion is calculated. It is assumed that
the time evolution goes from 0 to tf.
- order: the number of calculated terms in the Magnus expansion.
- A: the matrix of function that defines the linear ODE (dx/dt = A*x).
- numdims: the number of dimensions of the ODE.
"""
function solveOmega(tf, order, A, comm!, numdims)
	order_aux = order
	order = max(2, order)

	numtype = typeof(tf)
	S = createS(order, numdims, numtype)
	B = zeros(Rational{Int64}, order)
	for j = 1: order
		B[j] = bernoulli(j)//factorial(j)
	end
	B = convert(Array{numtype}, B)
	B = Tuple(B)

	Omega0 = zeros(numtype, numdims, order)
	tspan = (zero(tf), tf)
	temp1 = zeros(numtype, numdims)
	temp2 = zeros(numtype, numdims)
	temp3 = zeros(numtype, numdims)
	args = (order, A, comm!, S, B, temp1, temp2, temp3)
	prob = DE.ODEProblem(dOmega!, Omega0, tspan, args)
	sol = DE.solve(prob, DE.Vern9(), saveat=[tf], reltol=1e-11, abstol=1e-14,
				   maxiters = 10^6)

	if sol.retcode == :Success
		return sol(tf)[:, 1:order_aux]
	else
		error("Calculation of the Magnus series was unsuccessful.")
	end
end
"""
	createS(order, numdims, numtype)
Create the S array, which is an argument of the dOmega! function.
#Arguments:
-numtype: The type of the numbers used in problem.
See solveOmega documentation.
"""
function createS(order, numdims, numtype)
	S = [zeros(numtype, numdims, 1)]
	for n=2:order
		push!(S, zeros(numtype, numdims, n-1))
	end
	return S
end

### TEST ###
