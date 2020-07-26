using QuadGK
using FastClosures
#Harmonic correction in the rotating frame.
function WharmonicR(t, tf, coeffs)
	ε1, ε2, Δ = coeffs[1], coeffs[2], coeffs[3]
	cos2t, sin2t = cos(2*t), sin(2*t)
	g = (ε1*cos2t + ε2*sin2t)*(1 - cos(2*pi*t/tf))
	W = zeros(Float64, 3)
	W[1] = g*cos2t
	W[2] = g*sin2t
	W[3] = Δ/2
	return W
end
#Harmonic correction in the interaction frame.
function WharmonicI(t, tf, coeffs)
	ε1, ε2, Δ = coeffs[1], coeffs[2], coeffs[3]
	cos2t, sin2t = cos(2*t), sin(2*t)
	g = (ε1*cos2t + ε2*sin2t)*(1 - cos(2*pi*t/tf))
	Λt = Λ(t,tf)
	cosh2Λ, sinh2Λ = cosh(2*Λt), sinh(2*Λt)
	W = zeros(Float64, 3)
	W[1] = g*cos2t*cosh2Λ + (Δ/2 + g)*sinh2Λ
	W[2] = g*sin2t
	W[3] = g*cos2t*sinh2Λ + (Δ/2 + g)*cosh2Λ
	return W
end
#correction matrix for the harmonic correction.
function correction_matrix(tf)
	M = zeros(Float64, 3, 3)
	h(t) = 1 - cos(2*pi*t/tf)

	func11 = @closure t-> h(t)*((1 + cos(4*t))*cosh(2*Λ(t,tf))
								+ 2*cos(2*t)*sinh(2*Λ(t,tf)))/2
	M[1, 1], err = quadgk(func11, 0.0, tf, rtol=1e-10, atol=1e-14)
	func12 = @closure t-> h(t)*(sin(4*t)*cosh(2*Λ(t,tf))
								+ 2*sin(2*t)*sinh(2*Λ(t,tf)))/2
	M[1, 2], err = quadgk(func12, 0.0, tf, rtol=1e-10, atol=1e-14)
	func13 = @closure t-> sinh(2*Λ(t,tf))/2
	M[1, 3], err = quadgk(func13, 0.0, tf, rtol=1e-10, atol=1e-14)
	func21 = t-> h(t)*sin(4*t)/2
	M[2, 1], err = quadgk(func21, 0.0, tf, rtol=1e-10, atol=1e-14)
	func22 = t-> h(t)*(1 - cos(4*t))/2
	M[2, 2], err = quadgk(func22, 0.0, tf, rtol=1e-10, atol=1e-14)
	func31 = @closure t-> h(t)*((1 + cos(4*t))*sinh(2.0*Λ(t,tf))
								+ 2.0*cos(2*t)*cosh(2.0*Λ(t,tf)))/2
	M[3, 1], err = quadgk(func31, 0.0, tf, rtol=1e-10, atol=1e-14)
	func32 = @closure t-> h(t)*(sin(4*t)*sinh(2.0*Λ(t,tf))
								+ 2.0*sin(2*t)*cosh(2.0*Λ(t,tf)))/2
	M[3, 2], err = quadgk(func32, 0.0, tf, rtol=1e-10, atol=1e-14)
	func33 = @closure t-> cosh(2*Λ(t,tf))/2
	M[3, 3], err = quadgk(func33, 0.0, tf, rtol=1e-10, atol=1e-14)

	return M
end
#Calculate the coefficients of the harmonic correction.
function coefficients_W(Omega, M)
	ih_vec = Omega
	coeffs = inv(M)*ih_vec
	return coeffs
end

function get_coeffs(M, coeffs, order, tf)
	A = @closure (t-> -V_I(t, tf) .- WharmonicI(t, tf, coeffs))

    Ndims = 3
    Omegaf = solveOmega(tf, order, A, comm!, Ndims)

    vec = sum(Omegaf, dims=2)
    coeffs = coefficients_W(vec, M)
    return coeffs
end
function get_coeffs(M, order, tf)
    coeffs = zeros(3)
    for k = 1: order
		coeffs2 = get_coeffs(M, coeffs, order, tf)
        coeffs = coeffs .+ coeffs2
    end
    return coeffs
end
