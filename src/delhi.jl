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

    rename!(df, vcat([:year, :month, :date], features))
    df[!, :split] = [t <= num_train ? "train" : "test" for t in 1:size(df, 1)]
    df_train, df_test = groupby(df, "split")
    t_train = df_train[!, :date]'
    y_train = Matrix(df_train[!, features])'
    t_test = df_test[!, :date]'
    y_test = Matrix(df_test[!, features])'

    date_trans = FeatureNormalizer(t_train)
    feature_trans = FeatureNormalizer(y_train)

    return (vec(predict(date_trans, t_train)),
        predict(feature_trans, y_train),
        vec(predict(date_trans, t_test)),
        predict(feature_trans, y_test),
        date_trans,
        feature_trans)
end

end
