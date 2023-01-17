import Flux
using ProgressMeter: @showprogress
import Plots
#using Plots: scatter, plot, plot!, savefig

# create a Linear model struct: 
struct Linear
	W::Matrix{Float32}
	b::Vector{Float32}
end

# define what happens when you call the model with two integers
Linear(in::Int, out::Int) = Linear(Flux.glorot_uniform(out, in), zeros(Float32, out))

# define what happens when you call a Linear struct with a single Matrix{Float32}
(l::Linear)(data::Matrix{Float32}) = l.W * data .+ l.b

# make the backward propagation work.
Flux.@functor Linear

# Make random data with k dimensions
function make_k_data(k)
	n = 1000
	# create a (k,n) matrix
	data = rand(Float32, k, n)
	# define a target
	W = rand(1, k) 
	target = W * data  .+ rand(1)

	train = n-64
	test = n-63
	trainloader = Flux.DataLoader((data[:, 1:train], target[:, 1:train]), batchsize=64)
	testloader = Flux.DataLoader((data[:, test:end], target[:, test:end]), batchsize=64)
	trainloader, testloader
end

trainloader, testloader = make_k_data(10)

# create a MSE loss function
mse(ŷ, y) = sum((ŷ .- y).^2) / length(y)

function train(trainloader, k, lossfn, epochs) 
	model = Linear(k, 1) 
	optimizer = Flux.setup(Flux.Adam(1e-3), model) 

	losses = []
	@showprogress for epoch in 1:epochs
		for (x, y) in trainloader
			loss, grads = Flux.withgradient(model) do m
				yhat = m(x)
				lossfn(yhat, y)
			end
			Flux.update!(optimizer, model, grads[1])
			push!(losses, loss)
		end
	end
	model, losses
end

model, losses = train(trainloader, 10, mse, 100);
p1 = Plots.plot(losses)
Plots.savefig(p1, "losses.png")

x, y = first(testloader);
ŷ = model(x)
p2 = Plots.scatter(y[1, :], ŷ[1,:])
Plots.plot!(p2, [0,5], [0,5])
Plots.savefig(p2, "error.png")

			



