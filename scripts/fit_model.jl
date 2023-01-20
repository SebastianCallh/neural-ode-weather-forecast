using Random
using Dates
using Optimization
using Lux
using DiffEqFlux: NeuralODE, ADAMW, swish
using DifferentialEquations
using ComponentArrays
using BSON: @save, @load

include(joinpath("..", "src", "delhi.jl"))
include(joinpath("..", "src", "figures.jl"))

function neural_ode(t, data_dim)
    f = Lux.Chain(
        Lux.Dense(data_dim, 64, swish),
        Lux.Dense(64, 32, swish),
        Lux.Dense(32, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=t,
        abstol=1e-9, reltol=1e-9
    )
    
    rng = Random.default_rng()
    p, state = Lux.setup(rng, f)

    return node, ComponentArray(p), state
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

    θs, losses = ComponentArray[], Float32[]
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
rng = MersenneTwister(123)
df = Delhi.load()
plt_features = Delhi.plot_features(df)
savefig(plt_features, joinpath("plots", "features.svg"))

df_2016 = filter(x -> x.date < Date(2016, 1, 1), df)
plt_2016 = plot(
    df_2016.date,
    df_2016.meanpressure,
    title = "Mean pressure, before 2016",
    ylabel = Delhi.units[4],
    xlabel = "Time",
    color = 4,
    size = (600, 300),
    label = nothing,
    right_margin=5Plots.mm
)
savefig(plt_2016, joinpath("plots", "zoomed_pressure.svg"))

t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = Delhi.preprocess(df)

plt_split = plot(
    reshape(t_train, :), y_train',
    linewidth = 3, colors = 1:4,
    xlabel = "Normalized time", ylabel = "Normalized values",
    label = nothing, title = "Pre-processed data"
)
plot!(
    plt_split, reshape(t_test, :), y_test',
    linewidth = 3, linestyle = :dash,
    color = [1 2 3 4], label = nothing
)

plot!(
    plt_split, [0], [0], linewidth = 0,
    label = "Train", color = 1
)
plot!(
    plt_split, [0], [0], linewidth = 0,
    linestyle = :dash, label = "Test",
    color = 1
)
savefig(plt_split, joinpath("plots", "train_test_split.svg"))

obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
maxiters = 150
lr = 5e-3
θs, state, losses = train(t_train, y_train, obs_grid, maxiters, lr, rng, progress=true);
@save "artefacts/training_output.bson" θs losses

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

@info "Generating training animation..."
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
