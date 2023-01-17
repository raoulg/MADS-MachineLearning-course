using Base: @kwdef
import MLDatasets
import Flux
using ProgressMeter: @showprogress
using Flux: Chain, Conv, MaxPool, flatten, Dense, relu
using TensorBoardLogger: TBLogger
using Logging: with_logger
using Dates
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@kwdef mutable struct Args
    η::Float32 = 3e-4             ## learning rate
    λ::Float32 = 0.01                ## L2 regularizer param, implemented as weight decay
    batchsize::Int = 128      ## batch size
    epochs::Int = 3          ## number of epochs
    logdir::String = "logs/"   ## results path
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    train_loader = Flux.DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = Flux.DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

lossfn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
accuracy(ŷ, y) = sum(Flux.onecold(y) .== Flux.onecold(ŷ)) / size(y)[end]
round4(x) = round(x, digits=4)

function train(trainloader, lossfn, epochs) 
	model = LeNet5() 
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS")
    logger = TBLogger(args.logdir * timestamp)
	optimizer = Flux.setup(Flux.Adam(1e-3), model) 
    n = length(trainloader)

	@showprogress for epoch in 1:epochs
        losses = 0.0
        acc = 0.0
		for (x, y) in trainloader
			loss, grads = Flux.withgradient(model) do m
				ŷ = m(x)
                acc += accuracy(ŷ, y)
				lossfn(ŷ, y)
			end
            losses += loss

			Flux.update!(optimizer, model, grads[1])
		end

        with_logger(logger) do
            @info "train" loss = losses / n acc = acc / n
        end

	end
	model
end


args = Args()
trainloader, testloader = get_data(args)
trainedmodel = train(trainloader, lossfn, args.epochs)
