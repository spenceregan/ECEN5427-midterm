module NetworkDynamics
export Network, Asys, LFSolution, AsScheduled
using DataFrames, CSV, Statistics, LinearAlgebra
using CairoMakie

ω() = 2*π*60

abstract type NetworkBus end
abstract type GenBus <: NetworkBus end

mutable struct SlackBus <: GenBus
    V::Float64 # pu
    const θ::Float64 # radians
    P::Float64 # pu
    Q::Float64 # pu
    ∂P0::Float64 # constant current coefficient
    ∂Q0::Float64 # constant current coefficient
    ∂∂P0::Float64 # constant impedance coefficient
    ∂∂Q0::Float64 # constant impedance coefficient
end

SlackBus() = SlackBus(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

mutable struct PVBus <: GenBus
    V::Float64 # pu
    θ::Float64 # radians
    P::Float64 # pu
    Q::Float64 # pu
    ∂P0::Float64 # constant current coefficient
    ∂Q0::Float64 # constant current coefficient
    ∂∂P0::Float64 # constant impedance coefficient
    ∂∂Q0::Float64 # constant impedance coefficient
end
PVBus() = PVBus(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)



mutable struct PQBus <: NetworkBus
    V::Float64 # pu
    θ::Float64 # radians
    P::Float64 # pu
    Q::Float64 # pu
    ∂P0::Float64 # constant current coefficient
    ∂Q0::Float64 # constant current coefficient
    ∂∂P0::Float64 # constant impedance coefficient
    ∂∂Q0::Float64 # constant impedance coefficient
end
PQBus() = PQBus(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

complexV(bus::NetworkBus) = bus.V * exp(complex(0.0, bus.θ))
complexS(bus::NetworkBus) = complex(bus.P, bus.Q)
∂P_V(bus::NetworkBus) = bus.∂P0 + 2 * bus.∂∂P0 * bus.V
∂Q_V(bus::NetworkBus) = bus.∂Q0 + 2 * bus.∂∂Q0 * bus.V

mutable struct GenState
    E_q::Float64
    E_d::Float64
    δ::Float64
    ω::Float64
    E_fd::Float64
    R_f::Float64
    V_r::Float64
    Id::Float64
    Iq::Float64
    Vd::Float64
    Vq::Float64
end

mutable struct GenInput
    T_M::Float64
    Vref::Float64
end
GenInput() = GenInput(1.0, 1.0)

GenState() = GenState(1.0, 1.0, 0.0, ω(), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

struct Generator
    bus::Int
    id::Int
    H::Float64
    X_d::Float64
    Xpr_d::Float64
    X_q::Float64
    Xpr_q::Float64
    Tpr_d::Float64
    Tpr_q::Float64
    R_s::Float64
    D::Float64
    K_A::Float64
    T_A::Float64
    K_E::Float64
    T_E::Float64
    K_F::Float64
    T_F::Float64
    EF_C::Float64
    EF_exp::Float64
    x::GenState
    u::GenInput
end
S_E(g::Generator) = g.EF_C * exp(g.EF_exp * g.x.E_fd)
M(g::Generator) = 2*g.H/ω()
f_s(g::Generator) = (g.K_E + g.x.E_fd * g.EF_exp * S_E(g) + S_E(g)) / g.T_E

function xvec(gen::Generator)
    vec = [
        gen.x.δ,
        gen.x.ω,
        gen.x.E_q, 
        gen.x.E_d,
        gen.x.E_fd,
        gen.x.V_r,
        gen.x.R_f,
        gen.x.Id,
        gen.x.Iq,
        gen.x.Vd,
        gen.x.Vq
    ]
    return vec
end

function uvec(gen::Generator)
    vec = [gen.u.T_M, gen.u.Vref]
    return vec
end

struct Network
    Y::Array{Complex}
    buses::Vector{NetworkBus}
    gens::Vector{Generator}
    MVAbase::Float64
    ω::Float64 # angular frequency = 2πf
end

abstract type NetworkICs end
struct LFSolution <: NetworkICs end
struct AsScheduled <: NetworkICs end

function update_genstates_ss!(network::Network)
    V̄gss = V̄g(network)
    Īgss = Īg(network)
    g = 0
    for gen in network.gens
        g += 1
        gen.x.δ = angle(V̄gss[g] + complex(gen.R_s, gen.X_q) * Īgss[g])
    end

    Idss = Id(network)
    Iqss = Iq(network)
    Vdss = Vd(network)
    Vqss = Vq(network)

    g = 0
    for gen in network.gens
        g += 1
        gen.x.Id = Idss[g]
        gen.x.Iq = Iqss[g]
        gen.x.Vd = Vdss[g]
        gen.x.Vq = Vqss[g]
        gen.x.E_d = Vdss[g] + gen.R_s * Idss[g] - gen.Xpr_q * Iqss[g]
        gen.x.E_q = Vqss[g] + gen.R_s * Iqss[g] + gen.Xpr_d * Idss[g]
        gen.x.E_fd = gen.x.E_q + (gen.X_d - gen.Xpr_d) * Idss[g]
        gen.x.V_r = gen.x.E_fd * (gen.K_E + S_E(gen))
        gen.x.R_f = (gen.K_F / gen.T_F) * gen.x.E_fd
        gen.u.Vref = abs(V̄gss[g]) + (gen.x.V_r / gen.K_A)
        gen.x.ω = network.ω
        gen.u.T_M = gen.x.E_d * Idss[g] + gen.x.E_q * Iqss[g] + (gen.Xpr_q - gen.Xpr_d) * Idss[g] * Iqss[g]     
    end
end

function Network(
    Zdf::DataFrame, 
    busdf::DataFrame, 
    gendf::DataFrame, 
    init::LFSolution
    )
    n = nrow(busdf)
    YN = zeros(Complex, (n,n))
    buses = Vector{NetworkBus}(undef, n)
    gens = Vector{Generator}(undef, nrow(gendf))

    for i in 1:nrow(Zdf)
        busa = Zdf.busA[i]
        busb = Zdf.busB[i]
        r = Zdf.r[i]
        x = Zdf.x[i]
        z = complex(r,x)
        y = 1.0 / z
        yshunt = complex(0.0, Zdf.y_shunt[i])
        YN[busa, busb] = -y
        YN[busb, busa] = -y
        YN[busa, busa] += y + yshunt
        YN[busb, busb] += y + yshunt
    end

    for k in 1:n
        P = busdf.P[k]
        Q = busdf.Q[k]
        V = busdf.V[k]
        θ = 2 * π * (busdf.theta[k] / 360.0)
        ∂P = busdf.partialP0[k]
        ∂Q = busdf.partialQ0[k]
        ∂∂P = busdf.partial2P0[k]
        ∂∂Q = busdf.partial2Q0[k]
        if busdf.type[k] == "slack"
            bus = SlackBus(V, θ, P, Q, ∂P, ∂Q, ∂∂P, ∂∂Q)
        elseif busdf.type[k] == "PV"
            bus = PVBus(V, θ, P, Q, ∂P, ∂Q, ∂∂P, ∂∂Q)
        elseif busdf.type[k] == "PQ"
            bus = PQBus(V, θ, P, Q, ∂P, ∂Q, ∂∂P, ∂∂Q)
        else
            error("not a valid bus type listed in dataframe")
        end
        buses[k] = bus
    end

    for l in 1:nrow(gendf)
        params = (
            bus = gendf.bus[l],
            id = l,
            H = gendf.H[l],
            X_d = gendf.X_d[l],
            Xpr_d = gendf.Xpr_d[l],
            X_q = gendf.X_q[l],
            Xpr_q = gendf.Xpr_q[l],
            Tpr_d = gendf.Tpr_d[l],
            Tpr_q = gendf.Tpr_q[l],
            R_s = gendf.R_s[l],
            D = gendf.D[l],
            K_A = gendf.K_A[l],
            T_A = gendf.T_A[l],
            K_E = gendf.K_E[l],
            T_E = gendf.T_E[l],
            K_F = gendf.K_F[l],
            T_F = gendf.T_F[l],
            EF_C = gendf.EF_C[l],
            EF_exp = gendf.EF_exp[l],
            x = GenState(),
            u = GenInput()
        )
        gen = Generator(params...)
        gens[l] = gen
    end

    network = Network(YN, buses, gens, 100.0, ω())
    
    update_genstates_ss!(network)
    return network
end

function Network(
    Zdf::DataFrame, 
    busdf::DataFrame, 
    gendf::DataFrame, 
    init::AsScheduled
    )
    n = nrow(busdf)
    Y = zeros(ComplexF64, (n,n))
    buses = Vector{NetworkBus}(undef, n)
    gens = Vector{Generator}(undef, nrow(gendf))

    for i in 1:nrow(Zdf)
        busa = Zdf.busA[i]
        busb = Zdf.busB[i]
        r = Zdf.r[i]
        x = Zdf.x[i]
        z = complex(r,x)
        y = 1.0 / z
        yshunt = complex(0.0, Zdf.y_shunt[i])
        Y[busa, busb] = -y
        Y[busb, busa] = -y
        Y[busa, busa] += y + yshunt
        Y[busb, busb] += y + yshunt
    end

    for k in 1:n
        P = busdf.P[k]
        Q = busdf.Q[k]
        V = busdf.V[k]
        θ = 2 * π * (busdf.theta[k] / 360.0)
        ∂P = busdf.partialP0[k]
        ∂Q = busdf.partialQ0[k]
        ∂∂P = busdf.partial2P0[k]
        ∂∂Q = busdf.partial2Q0[k]
        if busdf.type[k] == "slack"
            bus = SlackBus(V, 0.0, P, Q, ∂P, ∂Q, ∂∂P, ∂∂Q)
        elseif busdf.type[k] == "PV"
            bus = PVBus(V, 0.0, P, 0.0, ∂P, ∂Q, ∂∂P, ∂∂Q)
        elseif busdf.type[k] == "PQ"
            bus = PQBus(1.0, 0.0, P, Q, ∂P, ∂Q, ∂∂P, ∂∂Q)
        else
            error("not a valid bus type listed in dataframe")
        end
        buses[k] = bus
    end

    for l in 1:nrow(gendf)
        params = (
            bus = gendf.bus[l],
            id = l,
            H = gendf.H[l],
            X_d = gendf.X_d[l],
            Xpr_d = gendf.Xpr_d[l],
            X_q = gendf.X_q[l],
            Xpr_q = gendf.Xpr_q[l],
            Tpr_d = gendf.Tpr_d[l],
            Tpr_q = gendf.Tpr_q[l],
            R_s = gendf.R_s[l],
            D = gendf.D[l],
            K_A = gendf.K_A[l],
            T_A = gendf.T_A[l],
            K_E = gendf.K_E[l],
            T_E = gendf.T_E[l],
            K_F = gendf.K_F[l],
            T_F = gendf.T_F[l],
            EF_C = gendf.EF_C[l],
            EF_exp = gendf.EF_exp[l],
            x = GenState(),
            u = GenInput()
        )
        gen = Generator(params...)
        gens[l] = gen
    end

    network = Network(Y, buses, gens, 100.0, ω())
    run_powerflow!(network)
    update_genstates_ss!(network)
    return network
end

allbuses(network::Network) = [i for i in 1:nbus(network)]
genbuses(network::Network) = [gen.bus for gen in network.gens]
loadbuses(network::Network) = setdiff(allbuses(network), genbuses(network))

xvec(network::Network) = reduce(vcat, [xvec(gen) for gen in network.gens])
uvec(network::Network) = reduce(vcat, [uvec(gen) for gen in network.gens])

Yangles(network::Network) = angle.(network.Y)

nbus(network::Network) = length(network.buses)

V̄(network::Network) = [complexV(bus) for bus in network.buses]
V(network::Network) = [bus.V for bus in network.buses]
V̄g(network::Network) = V̄(network)[genbuses(network)]
Vg(network::Network) = V(network)[genbuses(network)]
V̄l(network::Network) = V̄(network)[loadbuses(network)]
Vl(network::Network) = V(network)[loadbuses(network)]

S̄(ntwk::Network) = [complexS(ntwk.buses[i]) for i in 1:nbus(ntwk)]

θ(network::Network) = [bus.θ for bus in network.buses]
expθ(network::Network) = exp.(complex.(0, θ(network)))
θg(network::Network) = θ(network)[genbuses(network)]
expθg(network::Network) = exp.(complex.(0.0, θg(network)))

Ī(network::Network) = conj.(S̄(network) ./ V̄(network))
Īg(network::Network) = Ī(network)[genbuses(network)]
Ig(network::Network) = abs.(Īg)
Igγ(network::Network) = angle.(Īg)

function Ynd(network::Network)
    Ynd = copy(network.Y)
    Ynd[diagind(Ynd)] .= 0
    return Ynd
end
YV(network::Network) = network.Y * V̄(network)
YndV(network::Network) = Ynd(network) * V̄(network)
Yexpθ(network::Network) = network.Y * expθ(network)

δ(network::Network) = [gen.x.δ for gen in network.gens]
expδ(network::Network) = exp.(complex.(0.0, δ(network)))
δ_θ(network::Network) = δ(network) - θg(network)

Īdq(network::Network) = Īg(network) .* exp.(complex.(0.0, ((π/2) .- δ(network))))
Id(network::Network) = real.(Īdq(network))
Iq(network::Network) = imag.(Īdq(network))
Idq(network::Network) = vec(permutedims([Id(network) Iq(network)]))

V̄dq(network::Network) = V̄g(network) .* exp.(complex.(0.0, ((π/2) .- δ(network))))
Vd(network::Network) = real.(V̄dq(network))
Vq(network::Network) = imag.(V̄dq(network))

function ∂S_θ(network::Network)
    ∂S_θ = complex(0,-1) * (V̄(network) * V̄(network)' .* conj.(Ynd(network)))
    ∂S_θ += Diagonal(complex(0,1) * V̄(network) .* conj.(YndV(network)))
    return ∂S_θ
end
function ∂S_V(network::Network)
    ∂S_V = (V̄(network) * expθ(network)') .* conj.(network.Y)
    ∂S_V += Diagonal(expθ(network) .* conj.(YV(network)))
    return ∂S_V
end

∂Pload_V(network::Network) = [∂P_V(bus) for bus in network.buses]
∂Qload_V(network::Network) = [∂Q_V(bus) for bus in network.buses]
∂Sload_V(n::Network) = complex.(∂Pload_V(n), ∂Qload_V(n))

function A(network::Network)
    m = length(network.gens)
    A = zeros(Float64, (7*m,7*m))
    for gen in network.gens
        g = gen.id
        Ag = view(A, 7*g-6:7*g, 7*g-6:7*g)
        Ag[1,2] = 1.0
        Ag[2,2] = -gen.D / M(gen)
        Ag[2,3] = -gen.x.Iq / M(gen)
        Ag[2,4] = -gen.x.Id / M(gen)
        Ag[3,3] = -1 / gen.Tpr_d
        Ag[3,5] = 1 / gen.Tpr_d
        Ag[4,4] = -1 / gen.Tpr_q
        Ag[5,5] = f_s(gen)
        Ag[5,6] = 1 / gen.T_E
        Ag[6,5] = -(gen.K_A * gen.K_F) / (gen.T_A * gen.T_F)
        Ag[6,6] = -1 / gen.T_A
        Ag[6,7] = gen.K_A / gen.T_A
        Ag[7,5] = gen.K_F / (gen.T_F)^2 
        Ag[7,7] = -1 / gen.T_F
    end
    return A
end

function B1(network::Network)
    m = length(network.gens)
    B1 = zeros(Float64, (7*m,2*m))
    for gen in network.gens
        g = gen.id
        B1g = view(B1, 7*g-6:7*g, 2*g-1:2*g)
        B1g[2,1] = (gen.x.Iq * (gen.Xpr_d - gen.Xpr_q) - gen.x.E_d) / M(gen)
        B1g[2,2] = (gen.x.Id * (gen.Xpr_d - gen.Xpr_q) - gen.x.E_q) / M(gen)
        B1g[3,1] = -(gen.X_d - gen.Xpr_d) / gen.Tpr_d
        B1g[4,2] = (gen.X_q - gen.Xpr_q) / gen.Tpr_q
    end
    return B1
end

function B2(network::Network)
    m = length(network.gens)
    B2 = zeros(Float64, (7*m,2*m))
    for gen in network.gens
        g = gen.id
        B2g = view(B2, 7*g-6:7*g, 2*g-1:2*g)
        B2g[6,2] = -gen.K_A / gen.T_A
    end
    return B2
end

function E(network::Network)
    m = length(network.gens)
    E = zeros(Float64, (7*m,2*m))
    for gen in network.gens
        g = gen.id
        Eg = view(E, 7*g-6:7*g, 2*g-1:2*g)
        Eg[2,1] = 1 / M(gen)
        Eg[6,2] = gen.K_A / gen.T_A
    end
    return E
end

function C1(network::Network)
    m = length(network.gens)
    C1 = zeros(Float64, (2*m,7*m))
    γ = δ_θ(network)
    for gen in network.gens
        g = gen.id
        bus = gen.bus
        Vbus = network.buses[bus].V
        C1g = view(C1, 2*g-1:2*g, 7*g-6:7*g)
        C1g[1,1] = -Vbus * cos(γ[g])
        C1g[1,4] = 1
        C1g[2,1] = Vbus * sin(γ[g])
        C1g[2,3] = 1
    end
    return C1
end

function D1(network::Network)
    m = length(network.gens)
    D1 = zeros(Float64, (2*m,2*m))
    for gen in network.gens
        g = gen.id
        D1g = view(D1, 2*g-1:2*g, 2*g-1:2*g)
        D1g[1,1] = -gen.R_s
        D1g[1,2] = gen.Xpr_q
        D1g[2,1] = -gen.Xpr_d
        D1g[2,2] = -gen.R_s
    end
    return D1
end

function D2(network::Network)
    m = length(network.gens)
    D2 = zeros(Float64, (2*m,2*m))
    γ = δ_θ(network)
    for gen in network.gens
        g = gen.id
        bus = gen.bus
        Vbus = network.buses[bus].V
        D2g = view(D2, 2*g-1:2*g, 2*g-1:2*g)
        D2g[1,1] = Vbus * cos(γ[g])
        D2g[1,2] = -sin(γ[g])
        D2g[2,1] = -Vbus * sin(γ[g])
        D2g[2,2] = -cos(γ[g])
    end
    return D2
end

function C2(network::Network)
    m = length(network.gens)
    C2 = zeros(Float64, (2*m,7*m))
    γ = δ_θ(network)
    for gen in network.gens
        g = gen.id
        bus = gen.bus
        Vbus = network.buses[bus].V
        C2g = view(C2, 2*g-1:2*g, 7*g-6:7*g)
        C2g[1,1] = gen.x.Id * Vbus * cos(γ[g]) - gen.x.Iq * Vbus * sin(γ[g])
        C2g[2,1] = -gen.x.Id * Vbus * sin(γ[g]) - gen.x.Iq * Vbus * cos(γ[g])
    end
    return C2
end

function D3(network::Network)
    m = length(network.gens)
    D3 = zeros(Float64, (2*m,2*m))
    γ = δ_θ(network)
    for gen in network.gens
        g = gen.id
        bus = gen.bus
        Vbus = network.buses[bus].V
        D3g = view(D3, 2*g-1:2*g, 2*g-1:2g)
        D3g[1,1] = Vbus * sin(γ[g])
        D3g[1,2] = Vbus * cos(γ[g])
        D3g[2,1] = Vbus * cos(γ[g])
        D3g[2,2] = -Vbus * sin(γ[g])
    end
    return D3
end

function D4(network::Network)
    m = length(network.gens)
    D4 = zeros(Float64, (2*m,2*m))
    γ = δ_θ(network)
    ∂S_θik = -∂S_θ(network)
    ∂S_Vik = Diagonal(∂Sload_V(network)) - ∂S_V(network)
    for gen in network.gens
        i = gen.id
        D4ii = view(D4, 2*i-1:2*i, 2*i-1:2*i)
        genbusi = network.gens[i].bus
        Vbus = network.buses[genbusi].V
        D4ii[1,1] += gen.x.Iq * Vbus * sin(γ[i])
        D4ii[1,1] += -gen.x.Id * Vbus * cos(γ[i])
        D4ii[1,2] += gen.x.Id * sin(γ[i]) + gen.x.Iq * cos(γ[i])
        D4ii[2,1] += gen.x.Id * Vbus * sin(γ[i])
        D4ii[2,1] += gen.x.Iq * Vbus * cos(γ[i])
        D4ii[2,2] += gen.x.Id * cos(γ[i]) - gen.x.Iq * sin(γ[i])
        for k in 1:m
            D4ik = view(D4, 2*i-1:2*i, 2*k-1:2k)
            genbusk = network.gens[k].bus
            D4ik[1,1] += real(∂S_θik[genbusi, genbusk])
            D4ik[1,2] += real(∂S_Vik[genbusi, genbusk])
            D4ik[2,1] += imag(∂S_θik[genbusi, genbusk])
            D4ik[2,2] += imag(∂S_Vik[genbusi, genbusk])
        end
    end
    return D4
end

function D5(network::Network)
    m = length(network.gens)
    lm = length(loadbuses(network))
    lbuses = loadbuses(network)
    D5 = zeros(Float64, (2*m,2*lm))
    ∂S_θik = -∂S_θ(network)
    ∂S_Vik = Diagonal(∂Sload_V(network)) - ∂S_V(network)
    for i in 1:m
        genbusi = network.gens[i].bus
        for k in 1:lm
            D5ik = view(D5, 2*i-1:2*i, 2*k-1:2*k)
            loadbusk = lbuses[k]
            D5ik[1,1] += real(∂S_θik[genbusi, loadbusk])
            D5ik[1,2] += real(∂S_Vik[genbusi, loadbusk])
            D5ik[2,1] += imag(∂S_θik[genbusi, loadbusk])
            D5ik[2,2] += imag(∂S_Vik[genbusi, loadbusk])
        end
    end
    return D5
end

function D6(network::Network)
    m = length(network.gens)
    lbuses = loadbuses(network)
    lm = length(lbuses)
    D6 = zeros(Float64, (2*lm,2*m))
    ∂S_θik = -∂S_θ(network)
    ∂S_Vik = Diagonal(∂Sload_V(network)) - ∂S_V(network)
    for i in 1:lm
        loadbusi = lbuses[i]
        for k in 1:m
            D6ik = view(D6, 2*i-1:2*i, 2*k-1:2k)
            genbusk = network.gens[k].bus
            D6ik[1,1] += real(∂S_θik[loadbusi, genbusk])
            D6ik[1,2] += real(∂S_Vik[loadbusi, genbusk])
            D6ik[2,1] += imag(∂S_θik[loadbusi, genbusk])
            D6ik[2,2] += imag(∂S_Vik[loadbusi, genbusk])
        end
    end
    return D6
end

function D7(network::Network)
    lm = length(loadbuses(network))
    lbuses = loadbuses(network)
    D7 = zeros(Float64, (2*lm,2*lm))
    ∂S_θik = -∂S_θ(network)
    ∂S_Vik = Diagonal(∂Sload_V(network)) - ∂S_V(network)
    for i in 1:lm
        loadbusi = lbuses[i]
        for k in 1:lm
            loadbusk = loadbuses(network)[k]
            D7ik = view(D7, 2*i-1:2*i, 2*k-1:2k)
            D7ik[1,1] += real(∂S_θik[loadbusi, loadbusk])
            D7ik[1,2] += real(∂S_Vik[loadbusi, loadbusk])
            D7ik[2,1] += imag(∂S_θik[loadbusi, loadbusk])
            D7ik[2,2] += imag(∂S_Vik[loadbusi, loadbusk])
        end
    end
    return D7
end

K1(n::Network) = D4(n) - D3(n) * inv(D1(n)) * D2(n)
K2(n::Network) = C2(n) - D3(n) * inv(D1(n)) * C1(n)
Ā(n::Network) = A(n) - B1(n) * inv(D1(n)) * C1(n)
B̄(n::Network) = [(B2(n) - B1(n) * inv(D1(n)) * D2(n)) zeros(7*length(n.gens), 2*length(loadbuses(n)))]
C̄(n::Network) = [K2(n); zeros(2*length(loadbuses(n)), 7*length(n.gens))]
D̄(n::Network) = [K1(n) D5(n); D6(n) D7(n)]
Asys(n::Network) = Ā(n) - B̄(n) * inv(D̄(n)) * C̄(n)

function print_genstates(network::Network)
    convert = [180/π,1/(2*π),1,1,1,1,1,1,1,1,1]
    xrowlabs = ["δ", "ω(in Hz)", "E′q", "E′d", "Efd","V_R", "Rf", "Id", "Iq", "Vd", "Vq"]
    urowlabs = ["T_M", "Vref"]
    genstates = DataFrame(var = xrowlabs)
    geninputs = DataFrame(input = urowlabs)
    for i in 1:length(genbuses(network))
        gen = network.gens[i]
        bus = gen.bus
        genstates[!, string("gen", bus)] = convert .* xvec(gen)
        geninputs[!, string("gen", bus)] = uvec(gen)
    end
    display(genstates)
    display(geninputs)
    return genstates
end

nonslackbuses(n::Network) = [i for i in 1:length(n.buses) if !(isa(n.buses[i], SlackBus))]
pqbuses(n::Network) = [i for i in 1:length(n.buses) if isa(n.buses[i], PQBus)]

function Jlf(network::Network)
    pbuses = nonslackbuses(network)
    qbuses = pqbuses(network)
    ∂Sθ = ∂S_θ(network)
    ∂SV = ∂S_V(network)
    ∂P = [real.(∂S_θ(network)[pbuses, pbuses]) real.(∂S_V(network)[pbuses, qbuses])]
    ∂Q = [imag.(∂S_θ(network)[qbuses, pbuses]) imag.(∂S_V(network)[qbuses, qbuses])]
    Jlf = [∂P; ∂Q]
    return Jlf
end

function run_powerflow!(network::Network)
    tol = 10^(-6)
    pbuses = nonslackbuses(network)
    qbuses = pqbuses(network)
    slackbus = setdiff(collect(1:nbus(network)), pbuses)[1]
    Vbuses = setdiff(collect(1:nbus(network)), qbuses)
    ΔS = zeros(Float64, length(pbuses)+length(qbuses)) # [ΔP; ΔQ]
    ΔV = zeros(Float64, length(pbuses)+length(qbuses)) # [Δθ; ΔV]
    for iter in 1:5
        VI = V̄(network) .* conj.(YV(network))
        network.buses[slackbus].P = real(VI)[1]
        for i in Vbuses
            network.buses[i].Q = imag(VI)[i]
        end
        for i in 1:length(pbuses)
            pbus = pbuses[i]
            ΔS[i] = real(VI[pbus]) - network.buses[pbus].P
        end
        for k in 1:length(qbuses)
            qbus = qbuses[k]
            s = length(pbuses) + k
            ΔS[s] = imag(VI[qbus]) - network.buses[qbus].Q
        end

        if norm(ΔS) < tol
            break
        end
        ΔV = -inv(Jlf(network)) * ΔS

        for i in 1:length(pbuses)
            pbus = pbuses[i]
            network.buses[pbus].θ += ΔV[i]
        end
        for k in 1:length(qbuses)
            qbus = qbuses[k]
            s = length(pbuses) + k
            network.buses[qbus].V += ΔV[s]
        end
    end
end


function pflowcheck(network::Network)
    Serr = V̄(network) .* conj.(YV(network)) - S̄(network)
    RMSerr = sqrt(mean([abs2(err) for err in Serr]))
    return RMSerr
end

function participation_factors(network::Network)
    m = length(network.gens)
    v = eigen(Asys(network)).vectors
    w = abs.(inv(v))
    v = abs.(v)
    P = zeros(Float64, 7*m, 7*m)
    for i in 1:7*m
        for k in 1:7*m
            P[k,i] = (w[k,i]*v[k,i])/((w[:, i])' * v[:,i])
        end
    end
    return P
end


function main()
    root = dirname(@__FILE__)
    Zdfpath = joinpath(root, "network.csv")
    busespath = joinpath(root, "bustypes.csv")
    genspath = joinpath(root, "gendata.csv")
    Zdf = CSV.read(Zdfpath, DataFrame)
    busdf = CSV.read(busespath, DataFrame)
    gendf = CSV.read(genspath, DataFrame)
    mynetwork = Network(Zdf, busdf, gendf, LFSolution())
    eigenvals = eigen(Asys(mynetwork)).values
    pfactors = participation_factors(mynetwork)
    return eigenvals
end

end