using DifferentialEquations
using DiffEqFlux
using Plots
using Random

function neural_ode(t; data_dim=1)
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

function sol_anim(t_true_grid, u_true_grid,
                  t_pred_grid, u_pred_grid,
                  t_eval_grid, u_eval_grid)

    frames = vcat(
        1:length(t_true_grid),
        Int.(ones(100).*length(t_true_grid)) # pause frames
    )
    @animate for i in frames
        plt = plot(t_true_grid, u_true_grid, linewidth = 2,
                   title = "Solving an initial value problem",
                   label = "Target", legend = :topleft,
                   xlabel = "Time (t)", ylabel = "y(t)")

        plot!(plt, t_pred_grid[1:i], u_pred_grid[1:i],
              linewidth = 2, label = "Prediction")

        k = sum(t_eval_grid .< t_pred_grid[i])
        scatter!(plt, t_eval_grid[1:k], u_eval_grid[1:k],
                 color = 2, label = "Neural network evaluation")
    end
end

Random.seed!(1);
u0 = 1.0
a = 1.5
ex_f(u,p,t) = a*u
tspan = (0., 1.)
prob = ODEProblem(ex_f, u0, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-5, abstol = 1e-5)
t_eval_grid = sol.t
u_eval_grid = exp.(a.*sol.t)

interpol_grid = range(minimum(sol.t), maximum(sol.t), length = 100)
t_true_grid = range(tspan...; length=100)
u_true_grid = exp.(a*t_true_grid)

u_pred_grid = Array(solve(prob, Tsit5(),
                          saveat = interpol_grid,
                          reltol = 1e-2, abstol = 1e-2))

anim = sol_anim(interpol_grid, u_true_grid,
                interpol_grid, u_pred_grid,
                t_eval_grid, u_eval_grid)
gif(anim, "plots/neural-ode-explanation.gif", fps = 30)
