using Plots
include("delhi.jl")

function plot_pred(t, y, ŷ)
    plt = scatter(t, y, label="Observation")
    plot!(plt, t, ŷ, label="Prediction")
end

function plot_pred(t, y, t̂, ŷ; kwargs...)
    plot_params = zip(eachrow(y), eachrow(ŷ), Delhi.feature_names, Delhi.units)
    plts_preds = map(enumerate(plot_params)) do (i, (yᵢ, ŷᵢ, name, unit))
        plt = plot(
            t̂, ŷᵢ, label="Prediction", color=i, linewidth=3,
            legend=nothing, title=name; kwargs...
        )
        scatter!(
            plt, t, yᵢ, label="Observation",
            xlabel="Time", ylabel=unit,
            markersize=5, color=i
        )
    end
end

function plot_result(t, y, t̂, ŷ, loss, num_iters; kwargs...)
    plot_params = zip(eachrow(y), eachrow(ŷ), Delhi.feature_names, Delhi.units)
    plts_preds = plot_pred(t, y, t̂, ŷ; kwargs...)
    plot!(plts_preds[1], ylim=(10, 40), legend=(0.65, 1.0))
    plot!(plts_preds[2], ylim=(20, 100))
    plot!(plts_preds[3], ylim=(2, 12))
    plot!(plts_preds[4], ylim=(990, 1025))

    p_loss = plot(
        loss, label=nothing, linewidth=3,
        title="Loss", xlabel="Iterations",
        xlim=(0, num_iters)
    )
    plots = [plts_preds..., p_loss]
    plt = plot(plots..., layout=grid(length(plots), 1), size=(900, 900))
end

function animate_training(plot_frame, t_train, y_train, θs, losses, obs_grid; pause_for=300)
    obs_count = Dict(i - 1 => n for (i, n) in enumerate(obs_grid))
    is = [min(i, length(losses)) for i in 2:(length(losses) + pause_for)]
    @animate for i in is
        stage = Int(floor((i - 1) / length(losses) * length(obs_grid)))
        k = obs_count[stage]
        plt = plot_frame(t_train[1:k], y_train[:,1:k], θs[i], losses[1:i])
    end every 2
end

function plot_extrapolation(t_train, y_train, t_test, y_test, t̂, ŷ)
    plts = plot_pred(t_train, y_train, t̂, ŷ)
    for (i, (plt, y)) in enumerate(zip(plts, eachrow(y_test)))
        scatter!(plt, t_test, y, color=i, markerstrokecolor=:white, label="Test observation")
    end

    plot!(plts[1], ylim=(10, 40), legend=:topleft)
    plot!(plts[2], ylim=(20, 100))
    plot!(plts[3], ylim=(2, 12))
    plot!(plts[4], ylim=(990, 1025))
    plot(plts..., layout=grid(length(plts), 1), size=(900, 900))
end
