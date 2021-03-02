using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, Random, BSON

include("../src/delhi.jl")
include("../src/figures.jl")

function neural_ode(t, data_dim; saveat=t)
    f = FastChain(
        FastDense(data_dim, 64, swish),
        FastDense(64, 32, swish),
        FastDense(32, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=saveat, abstol=1e-9, reltol=1e-9
    )
end


function train_one_round(node, θ, y, opt, maxiters, y0=y[:, 1]; kwargs...)
    predict(θ) = Array(node(y0, θ))
    loss(θ) = Flux.mse(predict(θ), y)
    θ = θ === nothing ? node.p : θ
    res = DiffEqFlux.sciml_train(loss, θ, opt, maxiters=maxiters; kwargs...)
    res.minimizer
end

function train(t, y, obs_grid, θ=nothing, maxiters=150, lr=5e-3; kwargs...)
    log_results(θs, losses) =
        (θ, loss) -> begin
        push!(θs, copy(θ))
        push!(losses, loss)
        false
    end

    θs, losses = Vector{Float32}[], Float32[]
    for k in obs_grid
        node = neural_ode(t, size(y, 1))
        θ = train_one_round(
            node, θ, y, ADAMW(lr), maxiters;
            cb=log_results(θs, losses),
            kwargs...
        )
    end
    θs, losses
end

@info "Fitting model..."
Random.seed!(1)
t_train, y_train, t_test, y_test, t_trans, y_trans = Delhi.preprocess(Delhi.load())
obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
θs, losses = train(t_train, y_train, obs_grid; progress=true)
bson("artefacts/training_output.bson", params=θs, losses=losses)

@info "Generating training animation..."
predict(y0, t, θ) = begin
    node = neural_ode(t, length(y0))
    ŷ = Array(node(y0, θ))
end

function plot_pred(
    t_train, y_train, t_grid,
    rescale_t, rescale_y, num_iters, θ, loss, y0=y_train[:, 1]
)
    ŷ = predict(y0, t_grid, θ)
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
rescale_t(x) = t_trans.scale .* x .+ t_trans.offset
rescale_y(x) = y_trans.scale .* x .+ y_trans.offset
plot_frame(t, y, θ, loss) = plot_pred(
    t, y, t_train_grid, rescale_t, rescale_y, num_iters, θ, loss
)
anim = animate_training(plot_frame, t_train, y_train, θs, losses, obs_grid)
gif(anim, "plots/training.gif")

@info "Generating plot on extrapolation..."
t_grid = collect(range(minimum(t_train), maximum(t_test), length=500))
ŷ = predict(y_train[:,1], t_grid, θs[end])
plt_ext = plot_extrapolation(
    rescale_t(t_train),
    rescale_y(y_train),
    rescale_t(t_test),
    rescale_y(y_test),
    rescale_t(t_grid),
    rescale_y(ŷ)
)
savefig(plt_ext, "plots/extrapolation.svg")

@info "Done!"
