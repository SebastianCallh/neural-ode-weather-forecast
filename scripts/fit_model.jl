using Random
using Optimization
using Lux
using DiffEqFlux: NeuralODE, ADAMW, swish
using DifferentialEquations
using BSON: @save, @load

include("../src/delhi.jl")
include("../src/figures.jl")

function neural_ode(t, data_dim)
    f = Chain(
        Dense(data_dim, 64, swish),
        Dense(64, 32, swish),
        Dense(32, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=t,
        abstol=1e-9, reltol=1e-9
    )
    
    rng = Random.default_rng()
    p, state = Lux.setup(rng, f)

    return node, Lux.ComponentArray(p), state
end

function train_one_round(node, θ, state, y, opt, maxiters, rng, y0=y[:, 1]; kwargs...)
    predict(θ) = Array(node(y0, θ, state)[1])
    loss(θ) = sum(abs2, predict(θ) .- y)
    
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
    optprob = OptimizationProblem(optf, θ)
    res = solve(optprob, opt, maxiters=maxiters; kwargs...)
    res.minimizer, state
end

function train(t, y, obs_grid, maxiters, lr, rng, θ=nothing, state=nothing; kwargs...)
    log_results(θs, losses) =
        (θ, loss) -> begin
        push!(θs, copy(θ))
        push!(losses, loss)
        false
    end

    θs, losses = Lux.ComponentArray[], Float32[]
    for k in obs_grid
        node, θ_new, state_new = neural_ode(t, size(y, 1))
        if θ === nothing θ = θ_new end
        if state === nothing state = state_new end

        θ, state = train_one_round(
            node, θ, state, y, ADAMW(lr), maxiters, rng;
            callback=log_results(θs, losses),
            kwargs...
        )
    end
    θs, state, losses
end

@info "Fitting model..."
rng = Random.default_rng()
t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = Delhi.preprocess(Delhi.load())
obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
maxiters = 150
lr = 5e-3
θs, state, losses = train(t_train, y_train, obs_grid, maxiters, lr, rng, progress=true);
@save "artefacts/training_output.bson" θs losses

@info "Generating training animation..."
@load "artefacts/training_output.bson" θs losses

predict(y0, t, θ, state) = begin
    node, _, _ = neural_ode(t, length(y0))
    ŷ = Array(node(y0, θ, state)[1])
end

function plot_pred(
    t_train, y_train, t_grid,
    rescale_t, rescale_y, num_iters, θ, state, loss, y0=y_train[:, 1]
)
    ŷ = predict(y0, t_grid, θ, state)
    plt = plot_result(
        rescale_t(t_train),
        rescale_y(y_train),
        rescale_t(t_grid),
        rescale_y(ŷ),
        loss,
        num_iters
    )
end

num_iters = length(losses)
t_train_grid = collect(range(extrema(t_train)..., length=500))
rescale_t(x) = t_scale .* x .+ t_mean
rescale_y(x) = y_scale .* x .+ y_mean
plot_frame(t, y, θ, loss) = plot_pred(
    t, y, t_train_grid, rescale_t, rescale_y, num_iters, θ, state, loss
)
anim = animate_training(plot_frame, t_train, y_train, θs, losses, obs_grid);
gif(anim, "plots/training.gif")

@info "Generating extrapolation plot..."
t_grid = collect(range(minimum(t_train), maximum(t_test), length=500))
ŷ = predict(y_train[:,1], t_grid, θs[end], state)
plt_ext = plot_extrapolation(
    rescale_t(t_train),
    rescale_y(y_train),
    rescale_t(t_test),
    rescale_y(y_test),
    rescale_t(t_grid),
    rescale_y(ŷ)
);
savefig(plt_ext, "plots/extrapolation.svg")

@info "Done!"
