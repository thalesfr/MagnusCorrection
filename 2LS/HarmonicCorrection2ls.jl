#Functions for the Harmonic correction in 2ls.
using LinearAlgebra
using QuadGK
using FastClosures

function correction_matrix(tf, env_Δ, env_pulse1, env_pulse2)
	nΔ = length(env_Δ)
	npulse1, npulse2 = length(env_pulse1), length(env_pulse2)
	dims = nΔ + npulse1 + npulse2
    M = zeros(3, dims)
    for i = 1: nΔ
        h1 = env_Δ[i]
        func1 = @closure t-> h1(t/tf)*sin(2*c(t, tf))
        int_func1, err = quadgk(func1, 0.0, tf, rtol=1e-10, atol=1e-14)
        func2 = @closure t-> h1(t/tf)*cos(2*c(t, tf))
        int_func2, err = quadgk(func2, 0.0, tf, rtol=1e-10, atol=1e-14)
        M[2, i], M[3, i] = int_func1, int_func2
    end
    for i = 1: npulse1
        h2 = env_pulse1[i]
        func3 = @closure t-> h2(t/tf)*(1 + cos(2*t))/2
        int_func3, err = quadgk(func3, 0.0, tf, rtol=1e-10, atol=1e-14)
        func4 = @closure t-> -h2(t/tf)*sin(2*t)*cos(2*c(t, tf))/2
        int_func4, err = quadgk(func4, 0.0, tf, rtol=1e-10, atol=1e-14)
        func5 = @closure t-> h2(t/tf)*sin(2*t)*sin(2*c(t, tf))/2
        int_func5, err = quadgk(func5, 0.0, tf, rtol=1e-10, atol=1e-14)

        col = nΔ + i
        M[1, col], M[2, col], M[3, col] = int_func3, int_func4, int_func5
    end
    for i = 1: npulse2
        h3 = env_pulse2[i]
        func6 = @closure t-> h3(t/tf)*sin(2*t)/2.0
        int_func6, err = quadgk(func6, 0.0, tf, rtol=1e-10, atol=1e-14)
        func7 = @closure t-> -h3(t/tf)*(1 - cos(2*t))*cos(2*c(t, tf))/2
        int_func7, err = quadgk(func7, 0.0, tf, rtol=1e-10, atol=1e-14)
        func8 = @closure t-> h3(t/tf)*(1 - cos(2*t))*sin(2*c(t, tf))/2
        int_func8, err = quadgk(func8, 0.0, tf, rtol=1e-10, atol=1e-14)

        col = nΔ + npulse1 + i
        M[1, col], M[2, col], M[3, col] = int_func6, int_func7, int_func8
    end
    return M
end

function coefficients_W(Omega, M)
    ih_vec = Omega[1: 3]
    coeffs = pinv(M)*ih_vec
    return coeffs
end

function g(t, tf, coeffs, env_pulse1, env_pulse2)
	npulse1, npulse2 = length(env_pulse1), length(env_pulse2)
	cost, sint = cos(t), sin(t)
	gt = zero(t)
    for i = 1: npulse1
        h2 = env_pulse1[i]
        gt += coeffs[i]*h2(t/tf)*cost
    end
    for i = 1: npulse2
        h3 = env_pulse2[i]
        col = npulse1 + i
        gt += coeffs[col]*h3(t/tf)*sint
    end
	return gt
end
function Δ(t, tf, coeffs, env_Δ)
	nΔ = length(env_Δ)
	Δt = zero(t)
    for i = 1: nΔ
        h1 = env_Δ[i]
        Δt += coeffs[i]*h1(t/tf)
    end
	return Δt
end
function Wharmonic(t, tf, coeffs, env_Δ, env_pulse1, env_pulse2)
	Δt = Δ(t, tf, coeffs[1], env_Δ)
	gt = g(t, tf, coeffs[2], env_pulse1, env_pulse2)

    W = zeros(4)
    W[1] = gt*cos(t)
    W[2] = -gt*sin(t)
    W[3] = Δt
    return W
end

function Wharmonic_I(t, tf, coeffs, env_Δ, env_pulse1, env_pulse2)
	cost, sint = cos(t), sin(t)
	cosθ, sinθ = cos(2*c(t, tf)), sin(2*c(t, tf))
	Δt = Δ(t, tf, coeffs[1], env_Δ)
	gt = g(t, tf, coeffs[2], env_pulse1, env_pulse2)

	W_I = zeros(4)
	W_I[1] = gt*cost
	W_I[2] = Δt*sinθ - gt*sint*cosθ
	W_I[3] = Δt*cosθ + gt*sint*sinθ
	return W_I
end

function get_coeffs(M, coeffs, order, tf, env_Δ, env_pulse1, env_pulse2)
	nΔ = length(env_Δ)
	A = @closure (t-> - V_I(t, tf)
					  - Wharmonic_I(t, tf, coeffs, env_Δ, env_pulse1, env_pulse2))

    Ndims = 4
    Omegaf = solveOmega(tf, order, A, comm!, Ndims)

    vec = sum(Omegaf, dims=2)
    coeffs = coefficients_W(vec, M)
    return (coeffs[1: nΔ], coeffs[nΔ + 1: end])
end
function get_coeffs(M, order, tf, env_Δ, env_pulse1, env_pulse2)
    nΔ = length(env_Δ)
	npulse1, npulse2 = length(env_pulse1), length(env_pulse2)
    coeffs = (zeros(nΔ),  zeros(npulse1 + npulse2))
    for k = 1: order
		coeffs2 = get_coeffs(M, coeffs, k, tf, env_Δ, env_pulse1, env_pulse2)
        coeffs = coeffs .+ coeffs2
    end
    return coeffs
end

function U(tf, σ, coeffs, env_Δ, env_pulse1, env_pulse2)
	H = @closure (t-> H0(t, tf, true)
					  + Wharmonic(t, tf, coeffs, env_Δ, env_pulse1, env_pulse2))

    sol = solveU(H, tf, σ)
    return sol(tf)
end
