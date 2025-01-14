using ITensors
using Random
using LinearAlgebra
using MKL
using Pkg
using JSON
Pkg.activate("CT")
using CT
using Printf

using ArgParse
using Serialization
""" compute domain wall as a function of t"""

function random_int(L,lower_bound,upper_bound,seed=nothing)
    # lower_bound = 2^(L-1)
    # upper_bound = 2^L - 1
    if seed !== nothing
        rng = MersenneTwister(seed)
        return rand(rng, lower_bound:upper_bound)
    else
        return rand(lower_bound:upper_bound)
    end
end

function run_dw_t(L::Int,p_ctrl::Float64,p_proj::Float64,seed::Int)
    ct=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=true,store_vec=false,ancilla=0,xj=Set([0]),x0=1//2^L)
    print("x0: ", ct.x0)
    # x0=1//2^(L÷2+1)   # at the midpoint, without label
    # x0=1//2^L     # at k=1, with label x01
    # x0=random_int(L,2^(L-1),2^L - 1, seed)//2^L # at k=L, with label x12
    # x0=random_int(seed,0,2^L-1)//2^L # at random k, with label x00, here seed needs a redefinition, maybe can be seed_v
    i=L
    tf=(ct.ancilla ==0) ? 2*ct.L^2 : div(ct.L^2,2)
    for idx in 1:tf
        i=CT.random_control!(ct,i,p_ctrl,p_proj)
    end
    MI=CT.bipartite_mutual_information_self_average(ct,0)
    return Dict("MI"=>MI)
end


function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--p_ctrl"
        arg_type = Float64
        default = 0.0
        help = "control rate"
        "--p_proj"
        arg_type = Float64
        default = 0.0
        help = "projection rate"
        "--L", "-L"
        arg_type = Int
        default = 8
        help = "system size"
        "--seed", "-C"
        arg_type = Int
        default = 0
        help = "random seed for circuit-- unitary and position of projection"
        "--seed_m", "-m"
        arg_type = Int
        default = 0
        help = "random seed for measurement outcome"
    end
    return parse_args(s)
end

function main()
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    args = parse_my_args()
    results = run_dw_t(args["L"], args["p_ctrl"], args["p_proj"], args["seed"])

    filename = "MPS_(0,1)_L$(args["L"])_pctrl$(@sprintf("%.3f", args["p_ctrl"]))_pproj$(@sprintf("%.3f", args["p_proj"]))_s$(args["seed"])_MI1_8.json"
    data_to_serialize = merge(results, Dict("args" => args))
    json_data = JSON.json(data_to_serialize)
    open(filename, "w") do f
        write(f, json_data)
    end
end


function main_interactive(L::Int,p_ctrl::Float64,p_proj::Float64,seed::Int)
    start_time = time()

    # println("Uses threads: ",BLAS.get_num_threads())
    # println("Uses backends: ",BLAS.get_config())
    # args = parse_my_args()
    args=Dict("L"=>L,"p_ctrl"=>p_ctrl,"p_proj"=>p_proj,"seed"=>seed)
    filename = "MPS_(0,1)_L$(args["L"])_pctrl$(@sprintf("%.3f", args["p_ctrl"]))_pproj$(@sprintf("%.3f", args["p_proj"]))_s$(args["seed"])_MI1_8.json"
    
    results = run_dw_t(L, p_ctrl, p_proj, seed)

    data_to_serialize = merge(results, Dict("args" => args))
    json_data = JSON.json(data_to_serialize)
    open(filename, "w") do f
        write(f, json_data)
    end
    elapsed_time = time() - start_time
    println("p_ctrl: ", args["p_ctrl"], " p_proj: ", p_proj, " L: ", L, " seed: ", seed, )
    println("Execution time: ", elapsed_time, " s")
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end




# julia run_CT_MPS_C_m.jl --p_ctrl 0.5 --p_proj 0.0 --L 8 --seed 0 