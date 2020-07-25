using FastClosures
using SpecialFunctions
using StaticArrays

#Definite integral of the envelope function f (devided by 2).
function cosine_pulse_integral(t, tf, θ)
	x = t/tf
	π2 = convert(eltype(t), pi)*2
	ct = (θ/2)*(x .- sin.(π2*x)/π2)
	return ct
end
#Envelope function.
function cosine_pulse(t, tf, θ; derivative::Bool = false)
	x = t/tf
	π2 = convert(eltype(t), pi)*2
	if !derivative
		return (θ/tf)*(1 .- cos.(π2*x))
	else
		return (θ*π2/(tf*tf))*sin.(π2*x)
	end
end

const normalized_integral = erf(3/sqrt(2))
function gaussian_pulse(t, tf, θ; derivative::Bool = false)
	σ = tf/6
	μ = tf/2
	n = normalized_integral*σ*sqrt(2*pi)
	x = (t .- μ)/σ
	ft = exp.(-x.*x/2)
	if !derivative
		ft = (θ/n)*ft
	else
		ft = -(θ/(n*σ))*x.*ft
	end
	return ft
end

function gaussian_pulse_integral(t, tf, θ)
	σ = tf/6
	μ = tf/2
	if t < μ
		x = (μ - t)/(sqrt(2)*σ)
		s = (normalized_integral - erf(x))/2
	else
		x = (t - μ)/(sqrt(2)*σ)
		s = (erf(x) + normalized_integral)/2
	end
	return s*θ/(2*normalized_integral)
end

function pre_pulse_basis!(pulse_vector, t::Real, tf, npulse; derivative::Bool=false)
	k0 = 2*pi/tf
	if derivative == false
		for n = 1: npulse
			φ = k0*n*t
			pulse_vector[2*n - 1] = 1 - cos(φ)
			pulse_vector[2*n] = sin(φ)
		end
	else
		for n = 1: npulse
			kn = k0*n
			pulse_vector[2*n - 1] = kn*sin(kn*t)
			pulse_vector[2*n] = kn*cos(kn*t)
		end
	end
	return nothing
end

function pre_Δbasis!(Δbasis, t, tf, nΔ; derivative::Bool=false)
	Δbasis[1] = 1.0
	k0 = 2*pi/tf
	for n = 1: nΔ
		kn = k0*n
		Δbasis[2*n] = cos(kn*t)
		Δbasis[2*n + 1] = sin(kn*t)
	end
	return nothing
end

function fourier_basis(tf, nΔ, npulse; basis_derivative::Bool=false)
	Δvector = zeros(1 + 2*nΔ)
	Δbasis! = @closure (v, t)-> pre_Δbasis!(v, t, tf, nΔ; derivative=false)
	Δbasis = (function! =  Δbasis!, vector = Δvector)

	pulse_vector = zeros(2*npulse)
	pulse_basis! = @closure (v, t)-> pre_pulse_basis!(v, t, tf, npulse, derivative=false)
	pulse_basis = (function! =  pulse_basis!, vector = pulse_vector)
	basis = (pulse = pulse_basis, detuning = Δbasis)

	if basis_derivative == true
		dΔvector = zeros(1 + 2*nΔ)
		dΔbasis! = @closure (v, t)-> pre_Δbasis!(v, t, tf, nΔ; derivative=true)
		dΔbasis = (function! =  dΔbasis!, vector = dΔvector)

		dpulse_vector = zeros(2*npulse)
		dpulse_basis! = @closure (v, t)-> pre_pulse_basis!(v, t, tf, npulse; derivative=true)
		dpulse_basis = (function! =  dpulse_basis!, vector = dpulse_vector)
		basis_derivative = (pulse= dpulse_basis, detuning= dΔbasis)
		return basis, basis_derivative
	else
		return basis
	end
end
