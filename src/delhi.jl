module Delhi
using DataFrames, CSV, Dates, Statistics, Plots, MLDataUtils


features = [:meantemp, :humidity, :wind_speed, :meanpressure]
units = ["Celcius", "g/m³ of water", "km/h", "hPa"]
feature_names = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]

"""Loads the entire Delhi dataset into a single dataframe."""
function load()
    df = vcat(
        CSV.read(pwd() * "/data/DailyDelhiClimateTrain.csv", DataFrame),
        CSV.read(pwd() * "/data/DailyDelhiClimateTest.csv", DataFrame),
    )
end

"""Plots each feature as a time series."""
function plot_features(df)
    plots = map(enumerate(zip(features, feature_names, units))) do (i, (f, n, u))
        plot(df[:, :date], df[:, f],
             title=n, label=nothing,
             ylabel=u, size=(800, 600),
             color=i)
    end

    n = length(plots)
    plot(plots..., layout=(Int(n / 2), Int(n / 2)))
end


function preprocess(raw_df, num_train=20)
    raw_df[:,:year] = Float64.(year.(raw_df[:,:date]))
    raw_df[:,:month] = Float64.(month.(raw_df[:,:date]))
    df = combine(
        groupby(raw_df, [:year, :month]),
        :date => (d -> mean(year.(d)) .+ mean(month.(d)) ./ 12),
        :meantemp => mean,
        :humidity => mean,
        :wind_speed => mean,
        :meanpressure => mean,
        renamecols=false
    )

    t_and_y(df) = df[!, :date]', Matrix(df[!, features])'
    t_train, y_train = t_and_y(df[1:num_train,:])
    t_test, y_test = t_and_y(df[num_train+1:end,:])    
    t_trans = FeatureNormalizer(t_train)
    y_trans = FeatureNormalizer(y_train)

    return (
        vec(predict(t_trans, t_train)),
        predict(y_trans, y_train),
        vec(predict(t_trans, t_test)),
        predict(y_trans, y_test),
        t_trans,
        y_trans
    )
end

end
