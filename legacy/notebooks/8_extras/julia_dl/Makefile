.DEFAULT: run

run:
	julia --project 02_mnist.jl
	make tb

linear:
	julia --project 01_linear.jl

tb:
	poetry run tensorboard --logdir=logs