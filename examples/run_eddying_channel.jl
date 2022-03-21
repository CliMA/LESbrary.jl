using Oceananigans
using LESbrary.IdealizedExperiments: eddying_channel_simulation

simulation = eddying_channel_simulation(architecture=GPU())

run!(simulation)
