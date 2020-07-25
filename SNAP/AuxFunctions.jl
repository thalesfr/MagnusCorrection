
function comm!(commAB, A, B)
	N = div(size(A)[1], 4)
	for j = 1: N
		indice = 4*(j - 1)
		comm2ls!(commAB, A, B, indice)
	end
	return nothing
end

function comm2ls!(commAB, A, B, indice)
	i1, i2, i3, i4 = indice + 1, indice + 2, indice + 3, indice + 4
	commAB[i1] = (A[i2]*B[i3] - A[i3]*B[i2])*2
	commAB[i2] = (A[i3]*B[i1] - A[i1]*B[i3])*2
	commAB[i3] = (A[i1]*B[i2] - A[i2]*B[i1])*2
	commAB[i4] = 0
	return nothing
end
function comm2ls!(commAB, A, B)
	commAB[1] = (A[2]*B[3] - A[3]*B[2])*2
	commAB[2] = (A[3]*B[1] - A[1]*B[3])*2
	commAB[3] = (A[1]*B[2] - A[2]*B[1])*2
	commAB[4] = 0
	return nothing
end
#Decompose a generic 2x2 Hermitian matrix M in the Pauli basis.
function pauli_decomp(M)
	c = zeros(4)
	c[1] = real(M[1, 2])
	c[2] = imag(M[2, 1])
	c[3] = real(M[1, 1] - M[2, 2])/2
	c[4] = real(M[1, 1] + M[2, 2])/2
	return c
end

function tensor2matrix(tensor)
	N = size(tensor)[3]
	matrix = zeros(Complex{Float64}, 2*N, 2*N)
	for n=1:N
		matrix[2*(n - 1) + 1: 2*n, 2*(n - 1) + 1: 2*n] .= tensor[:, :, n]
	end
	return matrix
end

function vector2matrix(v::Array{<:Number,1}, σ)
	#matrix = zeros(Complex{Float64}, 2, 2)
	matrix = v[1].*σ[1] .+ v[2].*σ[2] .+ v[3].*σ[3] .+ v[4].*σ[4]
	return matrix
end
function vector2matrix2(v, σ)
	#matrix = zeros(Complex{Float64}, 2, 2)
	matrix = v[1]*σ[1] + v[2]*σ[2] + v[3]*σ[3] + v[4]*σ[4]
	return matrix
end

function vector2matrix!(matrix, v, σ)
	matrix .= v[1].*σ[1] .+ v[2].*σ[2] .+ v[3].*σ[3] .+ v[4].*σ[4]
	return nothing
end

function vector2matrix(v::Array{<:Number,2}, σ)
	N = size(v)[1]
	matrix = zeros(Complex{Float64}, 2*N, 2*N)
	vj = zeros(Complex{Float64}, 4)
	for j=1:N
		vj .= v[:,j]
		matrix[2*(j - 1) + 1: 2*j, 2*(j - 1) + 1: 2*j] .= vector2matrix(vj , σ)
	end
	return matrix
end

function vector2matrix!(matrix, v::Array{<:Number,2}, σ)
	N = size(v)[1]
	vj = zeros(Complex{Float64}, 4)
	for j=1:N
		vj .= v[:,j]
		matrix[2*(j - 1) + 1: 2*j, 2*(j - 1) + 1: 2*j] .= vector2matrix(vj , σ)
	end
	return nothing
end

function fidelity(U, U_ideal, dims)
	U_ideal_dag = adjoint(U_ideal)
	M = U_ideal_dag[1: dims, 1: dims]*U[1: dims, 1: dims]
	M_dag = adjoint(M)
	fidelity = (tr(M*M_dag) + abs(tr(M))^2)/(dims*(dims + 1))
	return real(fidelity)
end

function fidelity2(U, U_ideal, dims)
	U_ideal_dag = adjoint(U_ideal)
	M = U_ideal_dag[1: dims, 1: dims]*U[1: dims, 1: dims]
	fidelity = abs(tr(M))/dims
	return fidelity
end

function exp_matrix(v)
	exp = zeros(Complex{Float64}, 4)
	a = norm(v)
	if a > 100.0*eps()
		n = v/a
		sina, cosa = sin(a), cos(a)
		exp[1] = 1im*sina*n[1]
		exp[2] = 1im*sina*n[2]
		exp[3] = 1im*sina*n[3]
		exp[4] = cosa
	else
		exp[4] = 1
	end
	return exp
end

function exp_matrix(v, N)
	exp = zeros(Complex{Float64}, 4, N)
	vj = zeros(Complex{Float64}, 4)
	index = levels .+ 1
	for j=1:N
		vj .= v[:, j]
		exp[:, j] .= exp_matrix(vj)
	end
	return exp
end
#Calculate y.σ = exp[i v.σ] x.σ exp[-i v.σ].
#a.σ = a[1]*σ1 + a[2]*σ2 + a[3]*σ3 + a[4]*σ4
function unitary_tranformation(x, v)
	a = norm(v)
	if a > 100*eps()
		n = SA[v[1]/a, v[2]/a, v[3]/a]
		cosa, sina = cos(a), sin(a)
		y = x*cosa*cosa
		y13 = view(y, 1: 3)
		x13 = view(x, 1: 3)
		p = 2*dot(n, x13)
		y13 .= y13 .- sin(2*a).*cross(n, x13) .+ (sina*sina).*(p.*n .- x13)
	else
		y = x
	end
	return y
end
function unitary_tranformation!(x, v)
	a = norm(v)
	if a > 100*eps()
		n = (v[1]/a, v[2]/a, v[3]/a)
		cosa, sina = cos(a), sin(a)
		cosa2, sina2 = cosa*cosa, sina*sina
		sin2a = 2*sina*cosa#sin(2.0*a)

		nXx = (n[2]*x[3] - n[3]*x[2], n[3]*x[1] - n[1]*x[3], n[1]*x[2] - n[2]*x[1])
		p = 2*(n[1]*x[1] + n[2]*x[2] + n[3]*x[3])
		x[1] = x[1]*cosa2 - sin2a*nXx[1] + sina2*(p*n[1] - x[1])
		x[2] = x[2]*cosa2 - sin2a*nXx[2] + sina2*(p*n[2] - x[2])
		x[3] = x[3]*cosa2 - sin2a*nXx[3] + sina2*(p*n[3] - x[3])
		x[4] = x[4]*cosa2
	end
	return nothing
end
#Calculate the unitary tranformation in the Hilbert space formed by the outer
#product between the two level system space and the cavity space:
#<n| y_n |n> = exp[i v[n,:].σ] x[n,:].σ exp[-i v[n,:].σ]
function unitary_tranformation(x, v, N::Int)
	u = zeros(4, N)
	for j = 1: N
		xj = view(x, :, j)
		vj = view(v, :, j)
		u[:,j] .= unitary_tranformation(xj, vj)
	end
	return u
end
function unitary_tranformation!(x, v, N::Int)
	for j = 1: N
		xj = view(x, :, j)
		vj = view(v, :, j)
		unitary_tranformation!(xj, vj)
	end
	return nothing
end

#Returns the angular frequency vector of a DFT
function fftfreq(n::Integer, dt)
	if n%2 == 0
		v1 = collect(0: 1: div(n, 2) - 1)
		v2 = -collect(div(n, 2): -1: 1)
		v = 2*pi*[v1; v2]/(dt*n)
	else
		v1 = collect(0: 1: div(n - 1,2))
		v2 = -collect(div(n - 1, 2): -1: 1)
		v = 2*pi*[v1; v2]/(dt*n)
	end
	return v
end
