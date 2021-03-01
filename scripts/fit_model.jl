using Flux, DiffEqFlux, DifferentialEquations, DataFrames, Random
include("../src/delhi.jl")

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


function train_one_round(
    node, θ, y, y0, opt, maxiters; kwargs...
)
    predict(θ) = Array(node(y0, θ))
    loss(θ) = Flux.mse(predict(θ), y)

    θ = θ === nothing ? node.p : θ
    res = DiffEqFlux.sciml_train(
        loss, θ, opt,
        maxiters=maxiters;
        kwargs...
    )
    res.minimizer
end

function train(t, y, θ=nothing, maxiters=150, lr=1e-2; kwargs...)
    log_results(θs, losses) =
        (θ, loss) -> begin
        push!(θs, copy(θ))
        push!(losses, loss)
        false
    end

    θs, losses = Vector{Float32}[], Float32[]
    num_obs = 4:4:length(t)
    for k in num_obs
        node = neural_ode(t, size(y, 1))
        θ = train_one_round(
            node, θ, y, y[:, 1],
            ADAMW(lr), maxiters;
            cb=log_results(θs, losses),
            kwargs...
        )
    end
    θs, losses
end

Random.seed!(1)
df = Delhi.load() |> Delhi.preprocess
train_df, test_df = groupby(df, "split")
train_t = train_df[!, :date]
train_y = Matrix(train_df[!, Delhi.features])'
θs, losses = train(train_t, train_y; progress=true)
