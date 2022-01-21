using Test
using Printf
using Logging
using Oceananigans
using LESbrary

Logging.global_logger(OceananigansLogger())

architectures = [CPU()]

