using Oceananigans
using LESbrary.IdealizedExperiments: eddying_channel_simulation

simulation = eddying_channel_simulation(architecture=GPU(), max_momentum_flux=1e-4, max_buoyancy_flux=1e-8)
