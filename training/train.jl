# Train Garnet model to assign force field parameters
# Runs multithreaded on CPU, spawns training simulations after a few epochs to run on GPU
# See https://github.com/greener-group/garnet/tree/main/training for how to run this script
# Change submit_training_sims to launch an appropriate job on your system
# License is MIT

using HDF5
using GraphNeuralNetworks
using Flux
using Zygote
import Enzyme
using Polynomials
using BSON
using ChainRulesCore
using Molly
import Chemfiles
using CairoMakie
using SQLite
using DataFrames
using TimerOutputs
using Dates
using LinearAlgebra
using Random
using Statistics

function submit_training_sims(submit_dir, ff_xml_fp, log_fp, epoch_n)
    run(`sbatch --partition=gpu --gres=gpu:1 --time=4:0:0 --output=$log_fp --job-name=sim$epoch_n --wrap="/path/to/miniconda3/envs/openmm/bin/python sim_training.py $submit_dir $ff_xml_fp"`)
end

const T = Float32
const out_dir        = ARGS[1]
const start_training = (length(ARGS) >= 2 ? (ARGS[2] == "true") : true)
const verbose        = (length(ARGS) >= 3 ? (ARGS[3] == "true") : false)

const hdf5_files = ("SPICE-2.0.1.hdf5", "RNA-DIVERSE-OPENFF-DEFAULT.hdf5",
                    "RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5", "RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5")
const fp_features         = joinpath("features", "features.tsv")
const fp_features_maceoff = joinpath("features", "features_maceoff.tsv")
const fp_features_gems    = joinpath("features", "features_gems.tsv")
const fp_features_cond    = joinpath("features", "features_cond.tsv")
const fp_features_rna     = joinpath("features", "features_rna.tsv")
const rna_label_fp        = joinpath("features", "rna_label_to_key.tsv")
const fp_molecules_train      = joinpath("splits", "molecules_train.txt")
const fp_molecules_val        = joinpath("splits", "molecules_val.txt")
const fp_molecules_test       = joinpath("splits", "molecules_test.txt")
const fp_molecules_train_gems = joinpath("splits", "molecules_train_gems.txt")
const fp_molecules_val_gems   = joinpath("splits", "molecules_val_gems.txt")
const fp_molecules_test_gems  = joinpath("splits", "molecules_test_gems.txt")
const data_dir = "."
const maceoff_dir = joinpath(data_dir, "data_maceoff")
const gems_db = SQLite.DB(joinpath(data_dir, "crambin.db"))
const ref_traj_dir = joinpath("condensed_data", "trajs_ref")
const exp_data_dir = joinpath("condensed_data", "exp_data")
const learning_rate = 1e-4
const optimiser = Adam
const mse_loss = false
const dropout_dense = zero(T)
const dropout_gnn = dropout_dense
const train_on_fs_intra = one(T) # one(T) or zero(T)
const train_on_fs_inter = one(T)
const train_on_pe = one(T)
const train_on_charges = one(T)
const train_on_charge_reg = one(T)
const train_on_enth_vap = one(T)
const train_on_enth_mixing = one(T)
const train_on_J_coupling = one(T)
const train_on_chem_shifts = one(T)
const use_bonds = true
const use_angles = true
const use_proptor = true
const use_improptor = true
const use_vdw = true
const use_coul = true
const loss_weight_force = T(1e-3)
const loss_weight_force_inter = T(20.0)
const loss_weight_energy = T(1e-2)
const loss_weight_charge = T(100.0)
const loss_weight_enth_vap = T(0.2)
const loss_weight_enth_mixing = T(0.4)
const loss_weight_J_coupling = T(0.01)
const loss_weight_chem_shift = T(0.5)
const loss_weight_torsion_ks = T(1e-2)
const loss_weight_regularisation = T(1e-5)
const loss_weight_charge_reg = zero(T)
const water_h_zero_vdw = false
const dft_charge_weight = one(T)
const charge_sq_loss = false
const skip_water_charges = false
const loss_energy_first_epoch = 1
const use_ref_simulations = true
const training_sims_first_epoch = 6 # 0 to not run training sims
const loss_energy_max = 200 * loss_weight_energy
const grad_clamp_val = T(1e3)
const gcn_conv_layer = "SAGEConv"
const aggr_str = "mean"
const activation_str = "relu"
const n_layers_nn = 2
const n_layers_atom_embed = n_layers_nn # Min 2
const n_layers_pool       = n_layers_nn # Min 2
const n_layers_features   = n_layers_nn # Min 2
const dim_embed = 64
const dim_embed_atom, dim_embed_inter, nn_dim_atom = dim_embed, dim_embed, 4
const dim_hidden_dense, dim_hidden_gnn, dim_hidden_pairwise = 128, 128, 64
const n_atom_features_in = 27
const cond_sim_frames = 101:250  # 10 ps snapshots, 1 ns equilibration, 1.5 ns production
const prot_sim_frames = 501:3000 # 10 ps snapshots, 5 ns equilibration, 25  ns production
const n_molecules_val, n_molecules_test = 3_000, 3_000
const n_molecules_val_gems, n_molecules_test_gems = 100, 100
const n_frames_val_cond, n_frames_val_prot = 500, 500
const enth_vap_gas_n_samples = 10
const enth_mixing_temp = T(298.15) # K
const prot_reweighting_n_samples = 200 # 0 to use all (length(prot_sim_frames) - n_frames_val_prot)
const prot_reweighting_n_batches = 100
const n_epochs = 1_000
const n_minibatch = 256 # Pairs
const vdw_functional_form = "dexp"
const nonbonded_method = "pme" # none/cutoff/pme, only for condensed phase and ER systems
const dist_nb_cutoff = T(1.0) # nm
const mixing_function = "lb"
const bond_functional_form = "harmonic"
const angle_functional_form = "harmonic"
const n_proper_terms = 6
const n_improper_terms = 2
const improper_regularisation = false
const torsion_periodicities = ntuple(i -> i, 6)
const torsion_phases = ntuple(i -> i % 2 == 0 ? T(π) : zero(T), 6)
const bohr_to_nm = T(5.29177210903e-2)
const hartree_to_kJpmol = T(4.3597447222071 * 6.02214076e2)
const force_conversion = hartree_to_kJpmol / bohr_to_nm
const eVpÅ_to_kJpmolpnm = T(964.8533212331)
const max_force_norm = T(1e4) # kJ/mol
const boundary_inf = CubicBoundary(T(Inf))
const save_every_epoch = true
const to = TimerOutput()

inverse_sigmoid(x) = log(x / (1 - x))
const starting_weight14_vdw  = inverse_sigmoid(T(0.5))
const starting_weight14_coul = inverse_sigmoid(T(0.8333))

BLAS.set_num_threads(1)

const element_i_to_name = ["H", "Li", "B" , "C", "N" , "O" , "F", "Na", "Mg", "Si",
                           "P", "S" , "Cl", "K", "Ca", "Br", "I"]

const atomic_masses = [
    1.008 , 6.94        , 10.81, 12.011, 14.007 , 15.999, 18.998403163, 22.98976928, 24.305,
    28.085, 30.973761998, 32.06, 35.45 , 39.0983, 40.078, 79.904      , 126.90447  ,
]

const subset_exclusions = ()
const subset_n_repeats = Dict(
    # SPICE
    "SPICE Solvated Amino Acids Single Points Dataset v1.1" => 100, # 1,300   -> 130,000
    "SPICE Ion Pairs Single Points Dataset v1.2"            => 10 , # 1,426   -> 14,260
    "SPICE Dipeptides Single Points Dataset v1.3"           => 10 , # 33,850  -> 338,500
    "SPICE Amino Acid Ligand v1.0"                          => 2  , # 194,174 -> 388,348
    "SPICE Solvated PubChem Set 1 v1.0"                     => 20 , # 13,934  -> 278,680
    "SPICE Water Clusters v1.0"                             => 100, # 1,000   -> 100,000
    # Takaba2024 Espaloma
    "RNA Single Point Dataset v1.0"                         => 10 , # 8,560   -> 85,600
    "RNA Nucleoside Single Point Dataset v1.0"              => 10 , # 120     -> 1,200
    "RNA Trinucleotide Single Point Dataset v1.0"           => 10 , # 6,080   -> 60,800
    # Kovacs2023 MACE-OFF23
    "MACE-OFF water"                                        => 100, # 1,681   -> 168,100
    # GEMS
    "GEMS crambin"                                          => 100, # 5,140   -> 514,000
    # Condensed ΔHvap
    "vapourisation"                                         => 2  , # 2,250   -> 4,500
    # Condensed ΔHmix
    "mixing"                                                => 4  , # 600     -> 2,400
)
# Ensure these systems are in the training set
const train_systems = (
    "water",
    "rna_nucleoside_1", "rna_nucleoside_2", "rna_nucleoside_3", "rna_nucleoside_4",
)
const spice_hdf5_fp = joinpath(data_dir, hdf5_files[1])

# Ensure these systems are held out for testing
const condensed_test_systems = (
    "vapourisation_liquid_CC(=O)C", # Acetone
    "mixing_combined_CNCCO_O",
    "mixing_combined_CCC(C)=O_Nc1ccccc1",
)

if !isnothing(out_dir) && !isdir(out_dir)
    mkdir(out_dir)
    cp("train.jl", joinpath(out_dir, "train.jl"))
    mkdir(joinpath(out_dir, "ff_xml"))
    mkdir(joinpath(out_dir, "training_sims"))
    if save_every_epoch
        mkdir(joinpath(out_dir, "models"))
    end
end

function report(args...)
    print(args...)
    if !isnothing(out_dir)
        open(joinpath(out_dir, "training.log"), "a") do of
            print(of, args...)
        end
    end
end

const lines_features = vcat(readlines(fp_features), readlines(fp_features_rna),
                            readlines(fp_features_maceoff))
const mol_features = Dict(Pair(String.(split(line, "\t"; limit=2))...) for line in lines_features)
if isnothing(fp_molecules_train)
    const molecules_excl = shuffle(collect(filter(k -> !(k in train_systems), keys(mol_features))))
    const molecules_val = molecules_excl[1:n_molecules_val]
    const molecules_test = molecules_excl[(n_molecules_val+1):(n_molecules_val+n_molecules_test)]
    const molecules_train = vcat(molecules_excl[(n_molecules_val+n_molecules_test+1):end], train_systems...)
    # Can write these files
else
    const molecules_train = readlines(fp_molecules_train)
    const molecules_val   = readlines(fp_molecules_val)
    const molecules_test  = readlines(fp_molecules_test)
    @assert length(molecules_val ) == n_molecules_val
    @assert length(molecules_test) == n_molecules_test
    @assert all(x -> x in molecules_train, train_systems)
end

const mol_features_gems = Dict(Pair(String.(split(line, "\t"; limit=2))...)
                               for line in readlines(fp_features_gems)
                               if !contains(line, "valence issue") &&
                                  !contains(line, "GEMS total charge"))
if isnothing(fp_molecules_train_gems)
    gems_shuffle = shuffle(collect(keys(mol_features_gems)))
    const molecules_val_gems = gems_shuffle[1:n_molecules_val_gems]
    const molecules_test_gems = gems_shuffle[(n_molecules_val_gems+1):(n_molecules_val_gems+n_molecules_test_gems)]
    const molecules_train_gems = gems_shuffle[(n_molecules_val_gems+n_molecules_test_gems+1):end]
    # Can write these files
else
    const molecules_train_gems = readlines(fp_molecules_train_gems)
    const molecules_val_gems   = readlines(fp_molecules_val_gems)
    const molecules_test_gems  = readlines(fp_molecules_test_gems)
    @assert length(molecules_val_gems) == n_molecules_val_gems
    @assert length(molecules_test_gems) == n_molecules_test_gems
end

const mol_features_cond = Dict(Pair(String.(split(line, "\t"; limit=2))...)
                               for line in readlines(fp_features_cond))
const molecules_cond = Tuple{String, T, Int, Int}[]
for mol_id in keys(mol_features_cond)
    mol_id in condensed_test_systems && continue
    if startswith(mol_id, "vapourisation_liquid_")
        n_repeats = get(subset_n_repeats, "vapourisation", 1)
        for temp in T.(285:10:325)
            for frame_i in cond_sim_frames
                for repeat_i in 1:n_repeats
                    push!(molecules_cond, (mol_id, temp, frame_i, repeat_i))
                end
            end
        end
    elseif startswith(mol_id, "mixing_combined_")
        n_repeats = get(subset_n_repeats, "mixing", 1)
        for frame_i in cond_sim_frames
            for repeat_i in 1:n_repeats
                push!(molecules_cond, (mol_id, enth_mixing_temp, frame_i, repeat_i))
            end
        end
    end
end
shuffle!(molecules_cond)
# These are regenerated each run
const molecules_val_cond = molecules_cond[1:n_frames_val_cond]
const molecules_train_cond = molecules_cond[(n_frames_val_cond+1):end]

const enth_vap_exp_data = Dict{String, Polynomial{Float64, :x}}()
for mol_id in keys(mol_features_cond)
    if startswith(mol_id, "vapourisation_liquid_")
        smiles = split(mol_id, "_")[end]
        enth_vap_data_fp = joinpath(exp_data_dir, "enth_vap", "$smiles.txt")
        # Float32 gave bad fit
        enth_vap_exp_xs = parse.(Float64, getindex.(split.(readlines(enth_vap_data_fp)), 1))
        enth_vap_exp_ys = parse.(Float64, getindex.(split.(readlines(enth_vap_data_fp)), 2))
        enth_vap_exp_data[mol_id] = fit(enth_vap_exp_xs, enth_vap_exp_ys, 3)
    end
end

const enth_mixing_exp_data = Dict{String, T}()
for line in readlines(joinpath(exp_data_dir, "enth_mixing.tsv"))
    mol_id, enth_mixing = split(line, "\t")
    enth_mixing_exp_data[mol_id] = parse(T, enth_mixing)
end

struct ProteinData
    mol_names::Vector{String}
    mols_unique::Vector{Int}
    atom_mapping::Vector{Int}
    bond_mapping::Vector{Int}
    angle_mapping::Vector{Int}
    n_res::Int
end

const protein_data = Dict("gb3" => ProteinData(
    ["gb3", "O", "Na"],
    # These can be calculated from the full and reduced files
    [1, 2, 3016],
    vcat(1:862, repeat(863:865, 3014), fill(866, 2)),
    vcat(1:868, repeat(869:870, 3014)),
    vcat(1:1565, fill(1566, 3014)),
    56,
))
const mol_features_prot = Dict{String, String}()
for protein in keys(protein_data)
    mol_features_prot["protein_$protein"] = only(readlines(
                        joinpath("condensed_data", protein, "$protein.tsv")))
    mol_features_prot["protein_$(protein)_full"] = only(readlines(
                        joinpath("condensed_data", protein, "$(protein)_full.tsv")))
end

const frames_prot = shuffle(prot_sim_frames)
const frames_prot_val = frames_prot[1:n_frames_val_prot]
const frames_prot_train = frames_prot[(n_frames_val_prot+1):end]

struct JCouplingData{T}
    present::Bool
    J::T
    σ::T
    i::Int
    j::Int
    k::Int
    l::Int
end

function read_J_coupling_data(protein)
    inds_fp = joinpath("condensed_data", protein, "$(protein)_phi_indices.txt")
    J_fp = joinpath("condensed_data", protein, "$protein-scalar-couplings.dat")
    inds_lines = readlines(inds_fp)
    n_res = length(inds_lines)
    data = fill(JCouplingData(false, zero(T), zero(T), 0, 0, 0, 0), n_res)
    for line in readlines(J_fp)[2:end]
        J_atoms, resnum_str, _, J, σ = split(line)
        if J_atoms == "3j_hn_ha"
            resnum = parse(Int, resnum_str)
            i, j, k, l = parse.(Int, split(inds_lines[resnum]))
            data[resnum] = JCouplingData(true, parse(T, J), parse(T, σ), i, j, k, l)
        end
    end
    return data
end

const J_coupling_data = Dict(protein => read_J_coupling_data(protein)
                             for protein in keys(protein_data))

function read_chem_shift_data(protein)
    shifts_CA, shifts_C, shifts_N, shifts_H = T[], T[], T[], T[]
    n_res = protein_data[protein].n_res
    for line in readlines(joinpath("condensed_data", protein, "chem_shifts.txt"))
        cols = split(line)
        if cols[8] == "CA"
            push!(shifts_CA, parse(T, cols[11]))
        elseif cols[8] == "C" && parse(Int, cols[6]) < n_res # No C-terminal value
            push!(shifts_C, parse(T, cols[11]))
        elseif cols[8] == "N" && parse(Int, cols[6]) > 1 # No N-terminal value
            push!(shifts_N, parse(T, cols[11]))
        elseif cols[8] == "H" && parse(Int, cols[6]) > 1 # No N-terminal value
            push!(shifts_H, parse(T, cols[11]))
        end
    end
    @assert length(shifts_CA) == n_res
    @assert length(shifts_C)  == n_res - 1
    @assert length(shifts_N)  == n_res - 1
    @assert length(shifts_H)  == n_res - 1
    return vcat(shifts_CA, shifts_C, shifts_N, shifts_H)
end

const chem_shift_data = Dict(protein => read_chem_shift_data(protein)
                             for protein in keys(protein_data))

const mol_ids_xml = [split(line, "\t")[1] for line in readlines(fp_features_cond)
                     if any(startswith.(line, ("mixing_", "vapourisation_gas_")))]
for protein in keys(protein_data)
    push!(mol_ids_xml, "protein_$protein")
end

const rna_label_to_key = Dict(Pair(String.(split(line, "\t"))...)
                              for line in readlines(rna_label_fp))

struct XYZFile
    filepath::String
end

extract_n_confs(arr) = (length(arr) > 0 ? size(arr, 3) : 0)

function extract_hdf5_or_xyz(mol_id, hdf5_list)
    if startswith(mol_id, "maceoff_water")
        conf_i = parse(Int, split(mol_id, "_")[end])
        xyz = XYZFile(joinpath(maceoff_dir, "water", "conf_$conf_i.xyz"))
        subset = "MACE-OFF water"
        n_confs = 1
        return xyz, subset, n_confs
    elseif startswith(mol_id, "rna_diverse")
        mol_hdf5 = hdf5_list[2][rna_label_to_key[mol_id]]
        return mol_hdf5, mol_hdf5["subset"][1], extract_n_confs(mol_hdf5["dft_total_gradient"])
    elseif startswith(mol_id, "rna_nucleoside")
        mol_hdf5 = hdf5_list[3][rna_label_to_key[mol_id]]
        return mol_hdf5, mol_hdf5["subset"][1], extract_n_confs(mol_hdf5["dft_total_gradient"])
    elseif startswith(mol_id, "rna_trinucleotide")
        mol_hdf5 = hdf5_list[4][rna_label_to_key[mol_id]]
        return mol_hdf5, mol_hdf5["subset"][1], extract_n_confs(mol_hdf5["dft_total_gradient"])
    else
        mol_hdf5 = hdf5_list[1][mol_id]
        return mol_hdf5, mol_hdf5["subset"][1], extract_n_confs(mol_hdf5["dft_total_gradient"])
    end
end

function read_conformations(mol_ids, repeat_subsets=true)
    hdf5_list = [h5open(joinpath(data_dir, hdf5_file), "r") for hdf5_file in hdf5_files]
    mol_conf_pairs = Tuple{String, Int, Int, Int}[]
    for mol_id in mol_ids
        _, subset, n_confs = extract_hdf5_or_xyz(mol_id, hdf5_list)
        if !(subset in subset_exclusions) && n_confs > 0 # Some are empty
            if repeat_subsets
                n_repeats = get(subset_n_repeats, subset, 1)
            else
                n_repeats = 1
            end
            # Take random pairs of conformations, with 0 indicating no available pair
            #   due to an odd number of conformations
            conf_order = shuffle(1:n_confs)
            for ci in 1:2:n_confs
                conf_i = conf_order[ci]
                conf_i_p1 = (ci + 1 <= n_confs ? conf_order[ci + 1] : 0)
                for repeat_i in 1:n_repeats
                    push!(mol_conf_pairs, (mol_id, conf_i, conf_i_p1, repeat_i))
                end
            end
        end
    end
    close.(hdf5_list)
    return mol_conf_pairs
end

if vdw_functional_form in ("lj", "lj69")
    const n_vdw_atom_params = 2
    const global_params = [starting_weight14_vdw, starting_weight14_coul]
elseif vdw_functional_form in ("dexp", "buff")
    const n_vdw_atom_params = 2
    const global_params = [starting_weight14_vdw, starting_weight14_coul, zero(T), zero(T)]
elseif vdw_functional_form == "buck"
    const n_vdw_atom_params = 3
    const global_params = [starting_weight14_vdw, starting_weight14_coul]
elseif vdw_functional_form == "nn"
    training_sims_first_epoch == 0 || error("cannot run training simulations with vdw functional form nn")
    nonbonded_method == "none" || error("nonbonded method must be none with vdw functional form nn")
    const n_vdw_atom_params = nn_dim_atom
    const n_params_pairwise = (2 * nn_dim_atom + 3 + 1 + 1) * dim_hidden_pairwise
    const global_params = vcat(starting_weight14_vdw, T.(Flux.kaiming_uniform(n_params_pairwise)))
else
    error("unknown vdw functional form $vdw_functional_form")
end

if mixing_function == "lb"
    const σ_mixing = Molly.lorentz_σ_mixing
    const ϵ_mixing = Molly.geometric_ϵ_mixing
    const σ_string = "((sigma1 + sigma2) / 2)"
    const ϵ_string = "sqrt(epsilon1 * epsilon2)"
elseif mixing_function == "geom"
    const σ_mixing = Molly.geometric_σ_mixing
    const ϵ_mixing = Molly.geometric_ϵ_mixing
    const σ_string = "sqrt(sigma1 * sigma2)"
    const ϵ_string = "sqrt(epsilon1 * epsilon2)"
elseif mixing_function == "wh"
    const σ_mixing = Molly.waldman_hagler_σ_mixing
    const ϵ_mixing = Molly.waldman_hagler_ϵ_mixing
    const σ_string = "((sigma1^6 + sigma2^6) / 2) ^ (1/6)"
    const ϵ_string = "2 * sqrt(epsilon1 * epsilon2) * ((sigma1^3 * sigma2^3) / (sigma1^6 + sigma2^6))"
else
    error("unknown mixing function $mixing_function")
end

if !(nonbonded_method in ("none", "cutoff", "pme"))
    error("unknown nonbonded method $nonbonded_method")
end

if bond_functional_form == "harmonic"
    const n_bonded_params = 2
elseif bond_functional_form == "morse"
    const n_bonded_params = 3
else
    error("unknown bond functional form $bond_functional_form")
end

if angle_functional_form == "harmonic"
    const n_angle_params = 2
elseif angle_functional_form == "ub"
    const n_angle_params = 4
else
    error("unknown angle functional form $angle_functional_form")
end

if activation_str == "relu"
    const activation_gnn, activation_dense = relu, relu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif activation_str == "leakyrelu"
    const activation_gnn, activation_dense = leakyrelu, leakyrelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif activation_str == "gelu"
    const activation_gnn, activation_dense = gelu, gelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif activation_str == "swish"
    const activation_gnn, activation_dense = swish, swish
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
else
    error("unknown activation $activation_str")
end

if aggr_str == "mean"
    const aggr_gnn = mean
elseif aggr_str == "sum"
    const aggr_gnn = +
elseif aggr_str == "max"
    const aggr_gnn = max
else
    error("unknown aggregation function $aggr_str")
end

function gcn_conv(in_out_dims, activation=identity, init=Flux.glorot_uniform)
    if gcn_conv_layer == "GCNConv"
        return GCNConv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "GATv2Conv" # Zygote error
        return GATv2Conv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "SAGEConv"
        return SAGEConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif gcn_conv_layer == "GATConv" # Zygote error
        return GATConv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "GraphConv"
        return GraphConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif gcn_conv_layer == "ChebConv5" # Gives LAPACKException
        return ChebConv(in_out_dims, 5; init=init)
    else
        error("unknown graph convolutional layer $gcn_conv_layer")
    end
end

struct DoubleExponential{T, S, E, W} <: PairwiseInteraction
    α::T
    β::T
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
    dist_cutoff::T
end

function Base.zero(i::DoubleExponential{T, S, E, W}) where {T, S, E, W}
    return DoubleExponential(zero(T), zero(T), i.σ_mixing, i.ϵ_mixing, zero(W), zero(T))
end

function Base.:+(i1::DoubleExponential, i2::DoubleExponential)
    return DoubleExponential(i1.α + i2.α, i1.β + i2.β, i1.σ_mixing, i1.ϵ_mixing,
                             i1.weight_special + i2.weight_special, i1.dist_cutoff + i2.dist_cutoff)
end

Molly.use_neighbors(inter::DoubleExponential) = true

@inline function Molly.potential_energy(inter::DoubleExponential{T}, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...) where T
    if Molly.lj_zero_shortcut(atom_i, atom_j)
        return zero(T)
    end
    r = norm(vec_ij)
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    rm = σ * T(2^(1/6))
    α, β = inter.α, inter.β
    pe = ϵ * ((β * exp(α) / (α - β)) * exp(-α * r / rm) - (α * exp(β) / (α - β)) * exp(-β * r / rm))
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

@inline function Molly.force(inter::DoubleExponential{T}, vec_ij, atom_i, atom_j, force_units,
                             special, args...) where T
    if Molly.lj_zero_shortcut(atom_i, atom_j)
        return zero(SVector{3, T})
    end
    r = norm(vec_ij)
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    rm = σ * T(2^(1/6))
    α, β = inter.α, inter.β
    f = ϵ * (α * (β * exp(α) / (α - β)) * exp(-α * r / rm) - β * (α * exp(β) / (α - β)) * exp(-β * r / rm)) / rm
    fdr = f * normalize(vec_ij)
    if special
        return fdr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return fdr * (r <= inter.dist_cutoff)
    end
end

struct Buffered147{T, S, E, W} <: PairwiseInteraction
    δ::T
    γ::T
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
    dist_cutoff::T
end

function Base.zero(i::Buffered147{T, S, E, W}) where {T, S, E, W}
    return Buffered147(zero(T), zero(T), i.σ_mixing, i.ϵ_mixing, zero(W), zero(T))
end

function Base.:+(i1::Buffered147, i2::Buffered147)
    return Buffered147(i1.δ + i2.δ, i1.γ + i2.γ, i1.σ_mixing, i1.ϵ_mixing,
                       i1.weight_special + i2.weight_special, i1.dist_cutoff + i2.dist_cutoff)
end

Molly.use_neighbors(inter::Buffered147) = true

@inline function Molly.potential_energy(inter::Buffered147{T}, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...) where T
    if Molly.lj_zero_shortcut(atom_i, atom_j)
        return zero(T)
    end
    r = norm(vec_ij)
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    δ, γ = inter.δ, inter.γ
    rm = σ * T(2^(1/6))
    r_div_rm = r / rm
    pe = ϵ * ((1 + δ) / (r_div_rm + δ))^7 * (((1 + γ) / (r_div_rm^7 + γ)) - 2)
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

@inline function Molly.force(inter::Buffered147{T}, vec_ij, atom_i, atom_j, force_units,
                             special, args...) where T
    if Molly.lj_zero_shortcut(atom_i, atom_j)
        return zero(SVector{3, T})
    end
    r = norm(vec_ij)
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    δ, γ = inter.δ, inter.γ
    rm = σ * T(2^(1/6))
    r_div_rm = r / rm
    r_div_rm_6 = r_div_rm^6
    r_div_rm_7 = r_div_rm_6 * r_div_rm
    γ7_term = (1 + γ) / (r_div_rm_7 + γ)
    f = (7ϵ / rm) * ((1 + δ) / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
    fdr = f * normalize(vec_ij)
    if special
        return fdr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return fdr * (r <= inter.dist_cutoff)
    end
end

struct BuckinghamAtom{T}
    index::Int
    atom_type::Int
    mass::T
    charge::T
    A::T
    B::T
    C::T
end

function Base.zero(::BuckinghamAtom{T}) where T
    z = zero(T)
    return BuckinghamAtom(0, 0, z, z, z, z, z)
end

function Base.:+(a1::BuckinghamAtom, a2::BuckinghamAtom)
    return BuckinghamAtom(0, 0, a1.mass + a2.mass, a1.charge + a2.charge,
                          a1.A + a2.A, a1.B + a2.B, a1.C + a2.C)
end

struct Buckingham{T, W} <: PairwiseInteraction
    weight_special::W
    dist_cutoff::T
end

Base.zero(::Buckingham{T, W}) where {T, W} = Buckingham(zero(W), zero(T))

Base.:+(i1::Buckingham, i2::Buckingham) = Buckingham(i1.weight_special + i2.weight_special,
                                                     i1.dist_cutoff + i2.dist_cutoff)

Molly.use_neighbors(::Buckingham) = true

@inline function Molly.potential_energy(inter::Buckingham{T}, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...) where T
    if Molly.buckingham_zero_shortcut(atom_i, atom_j)
        return zero(T)
    end
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    A = sqrt(atom_i.A * atom_j.A)
    B = 2 / (inv(atom_i.B) + inv(atom_j.B))
    C = (atom_i.C + atom_j.C) / 2
    repulsion = T(0.1) * (T(0.2) / r)^12
    pe = repulsion + A * (exp(-B * r) - (C / r)^6)
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

@inline function Molly.force(inter::Buckingham{T}, vec_ij, atom_i, atom_j, force_units,
                             special, args...) where T
    if Molly.buckingham_zero_shortcut(atom_i, atom_j)
        return zero(SVector{3, T})
    end
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    A = sqrt(atom_i.A * atom_j.A)
    B = 2 / (inv(atom_i.B) + inv(atom_j.B))
    C = (atom_i.C + atom_j.C) / 2
    f_repulsion = T(1.2) * (T(0.2) / r)^12 / r2
    fdr = (f_repulsion + A * (B * exp(-B * r) / r - 6 * (C / r)^6 / r2)) * vec_ij
    if special
        return fdr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return fdr * (r <= inter.dist_cutoff)
    end
end

struct NNAtom{T}
    index::Int
    atom_type::Int
    mass::T
    charge::T
    params::Vector{T}
end

function Base.zero(a::NNAtom{T}) where T
    return NNAtom(0, 0, zero(T), zero(T), zero(a.params))
end

function Base.:+(a1::NNAtom, a2::NNAtom)
    return NNAtom(0, 0, a1.mass + a2.mass, a1.charge + a2.charge, a1.params .+ a2.params)
end

struct NNPairwise{T, W} <: PairwiseInteraction
    params::Vector{T}
    weight_special::W
    dist_cutoff::T
end

Base.zero(i::NNPairwise{T, W}) where {T, W} = NNPairwise(zero(i.params), zero(W), zero(T))

Base.:+(i1::NNPairwise, i2::NNPairwise) = NNPairwise(
    i1.params .+ i2.params,
    i1.weight_special + i2.weight_special,
    i1.dist_cutoff + i2.dist_cutoff,
)

Molly.use_neighbors(::NNPairwise) = true

function mvp!(o, a, b)
    @assert size(a, 2) == length(b)
    for i in axes(a, 1)
        s = zero(T)
        for j in axes(a, 2)
            s += a[i, j] * b[j]
        end
        o[i] = s
    end
    return o
end

@inline function Molly.potential_energy(inter::NNPairwise{T}, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...) where T
    r = sqrt(sum(abs2, vec_ij))
    @views if r <= inter.dist_cutoff
        inv_r = inv(r)
        inputs = vcat(atom_i.charge, atom_i.params, atom_j.charge, atom_j.params, inv_r)
        w1 = reshape(inter.params[1:(end-2*dim_hidden_pairwise)], dim_hidden_pairwise, length(inputs))
        b1 = inter.params[(end-2*dim_hidden_pairwise+1):(end-dim_hidden_pairwise)]
        w2 = reshape(inter.params[(end-dim_hidden_pairwise+1):end], 1, dim_hidden_pairwise)
        hl, out = zeros(T, size(w1, 1)), zeros(T, 1)
        mvp!(hl, w1, inputs)
        # Changing this activation also requires changing the force algorithm
        hl .= relu.(hl .+ b1)
        mvp!(out, w2, hl)
        pe_a = only(out)
        inputs[1] = atom_j.charge
        inputs[2:(nn_dim_atom + 1)] .= atom_j.params
        inputs[nn_dim_atom + 2] = atom_i.charge
        inputs[(nn_dim_atom + 3):(nn_dim_atom + 6)] .= atom_i.params
        mvp!(hl, w1, inputs)
        hl .= relu.(hl .+ b1)
        mvp!(out, w2, hl)
        pe_b = only(out)
        pe = (pe_a + pe_b) / 2
        if special
            return pe * inter.weight_special
        else
            return pe
        end
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::NNPairwise{T}, vec_ij, atom_i, atom_j, force_units,
                             special, args...) where T
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    @views if r <= inter.dist_cutoff
        inv_r = inv(r)
        inputs = vcat(atom_i.charge, atom_i.params, atom_j.charge, atom_j.params, inv_r)
        w1 = reshape(inter.params[1:(end-2*dim_hidden_pairwise)], dim_hidden_pairwise, length(inputs))
        b1 = inter.params[(end-2*dim_hidden_pairwise+1):(end-dim_hidden_pairwise)]
        w2_flat = inter.params[(end-dim_hidden_pairwise+1):end]
        hl_unact = zeros(T, size(w1, 1))
        mvp!(hl_unact, w1, inputs)
        hl_unact .= hl_unact .+ b1
        dE_dinvr = zero(T)
        @inbounds for i in 1:dim_hidden_pairwise
            if hl_unact[i] > zero(T)
                dE_dinvr += w2_flat[i] * w1[i, end]
            end
        end
        inputs[1] = atom_j.charge
        inputs[2:(nn_dim_atom + 1)] .= atom_j.params
        inputs[nn_dim_atom + 2] = atom_i.charge
        inputs[(nn_dim_atom + 3):(nn_dim_atom + 6)] .= atom_i.params
        mvp!(hl_unact, w1, inputs)
        hl_unact .= hl_unact .+ b1
        @inbounds for i in 1:dim_hidden_pairwise
            if hl_unact[i] > zero(T)
                dE_dinvr += w2_flat[i] * w1[i, end]
            end
        end
        dE_r = dE_dinvr * -inv(r2) / 2
        fdr = (-dE_r / r) * vec_ij
        if special
            return fdr * inter.weight_special
        else
            return fdr
        end
    else
        return zero(SVector{3, T})
    end
end

function generate_gnn_layers(n_layers)
    layers = []
    if dropout_gnn > 0
        push!(layers, Dropout(dropout_gnn))
    end
    for _ in 1:n_layers
        push!(layers, gcn_conv(dim_hidden_gnn => dim_hidden_gnn, activation_gnn, init_gnn))
        if dropout_gnn > 0
            push!(layers, Dropout(dropout_gnn))
        end
    end
    return layers
end

function generate_dense_layers(n_layers)
    layers = []
    if dropout_dense > 0
        push!(layers, Dropout(dropout_dense))
    end
    for _ in 1:n_layers
        push!(layers, Dense(dim_hidden_dense => dim_hidden_dense, activation_dense;
                            init=init_dense))
        if dropout_dense > 0
            push!(layers, Dropout(dropout_dense))
        end
    end
    return layers
end

model_atom_embed = GNNChain(
    gcn_conv(n_atom_features_in => dim_hidden_gnn, activation_gnn, init_gnn),
    generate_gnn_layers(n_layers_atom_embed - 2)...,
    gcn_conv(dim_hidden_gnn => dim_embed_atom),
)

model_bond_pool = Chain(
    Dense(dim_embed_atom * 2 => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_pool - 2)...,
    Dense(dim_hidden_dense => dim_embed_inter),
)

model_angle_pool = Chain(
    Dense(dim_embed_atom * 3 => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_pool - 2)...,
    Dense(dim_hidden_dense => dim_embed_inter),
)

model_proper_pool = Chain(
    Dense(dim_embed_atom * 4 => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_pool - 2)...,
    Dense(dim_hidden_dense => dim_embed_inter),
)

model_improper_pool = Chain(
    Dense(dim_embed_atom * 4 => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_pool - 2)...,
    Dense(dim_hidden_dense => dim_embed_inter),
)

model_atom_features = Chain(
    Dense(dim_embed_atom => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_features - 2)...,
    Dense(dim_hidden_dense => 2 + n_vdw_atom_params), # charge electronegativity, charge hardness, vdw params
)

model_bond_features = Chain(
    Dense(dim_embed_inter => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_features - 2)...,
    Dense(dim_hidden_dense => n_bonded_params), # k1, k2 (unless morse, then k1, k2, a)
)

model_angle_features = Chain(
    Dense(dim_embed_inter => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_features - 2)...,
    Dense(dim_hidden_dense => n_angle_params), # k1, k2 (unless ub, then k1a, k2a, k1b, k2b)
)

model_proper_features = Chain(
    Dense(dim_embed_inter => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_features - 2)...,
    Dense(dim_hidden_dense => n_proper_terms), # k for periodicity 1 to n_proper_terms
)

model_improper_features = Chain(
    Dense(dim_embed_inter => dim_hidden_dense, activation_dense; init=init_dense),
    generate_dense_layers(n_layers_features - 2)...,
    Dense(dim_hidden_dense => n_improper_terms), # k for periodicity 1 to n_improper_terms
)

struct GlobalParams{T}
    params::Vector{T}
end
(model::GlobalParams)() = model.params

model_global_params = GlobalParams(global_params)

models = [model_atom_embed, model_bond_pool, model_angle_pool, model_proper_pool,
          model_improper_pool, model_atom_features, model_bond_features, model_angle_features,
          model_proper_features, model_improper_features, model_global_params]

transform_lj_σ(x) = sigmoid(x) * T(0.45) + T(0.05) # 0.05 nm -> 0.5 nm

function transform_lj_ϵ(x, water_h)
    # 0.02 kJ/mol -> 1.5 kJ/mol
    return (sigmoid(x) * T(1.48) + T(0.02)) * !(water_h_zero_vdw && water_h)
end

transform_dexp_α(x) = sigmoid(x) * T(6.0) + T(9.5) # 9.5 -> 15.5
transform_dexp_β(x) = sigmoid(x) * T(4.0) + T(2.0) # 2.0 -> 6.0

transform_buff_δ(x) = sigmoid(x) * T(0.08) + T(0.07) # 0.07 -> 0.15
transform_buff_γ(x) = sigmoid(x) * T(0.16) + T(0.08) # 0.08 -> 0.24

function transform_buck_A(x, water_h)
    # 100_000 kJ/mol -> 900_000 kJ/mol
    return (sigmoid(x) * T(800_000.0) + T(100_000.0)) * !(water_h_zero_vdw && water_h)
end

transform_buck_B(x) = sigmoid(x) * T(40.0) + T(10.0) # 10 nm^-1 -> 50 nm^-1

function transform_buck_C(x, water_h)
    # With modified functional form this doesn't have to be set to zero for water Hs
    # 0.025 nm -> 0.075 nm
    return (sigmoid(x) * T(0.05) + T(0.025)) * !(water_h_zero_vdw && water_h)
end

transform_bond_k(  k1, k2) = max(k1 + k2, zero(T))
transform_angle_k( k1, k2) = max(k1 + k2, zero(T))
transform_bond_r0( k1, k2) = max((k1 * T(0.05)       + k2 * T(0.3)         ) / (k1 + k2), zero(T))
transform_angle_θ0(k1, k2) = max((k1 * T(deg2rad(0)) + k2 * T(deg2rad(180))) / (k1 + k2), zero(T))

transform_morse_a(a) = max(a, zero(T))

transform_ub_r0(k1, k2) = max((k1 * T(0.1) + k2 * T(0.6)) / (k1 + k2), zero(T))

function find_water_hs(adj_list, elements)
    Hs = falses(length(adj_list))
    # Safe to not set when water_h_zero_vdw is false
    if water_h_zero_vdw
        H_ind = findfirst(isequal("H"), element_i_to_name)
        O_ind = findfirst(isequal("O"), element_i_to_name)
        for (i, al) in enumerate(adj_list)
            if length(al) == 2 && elements[i] == H_ind
                bonded_i = al[2]
                bonded_adj = adj_list[bonded_i]
                if elements[bonded_i] == O_ind && length(bonded_adj) == 3 &&
                            elements[bonded_adj[2]] == H_ind && elements[bonded_adj[2]] == H_ind
                    Hs[i] = true
                end
            end
        end
    end
    return Hs
end

function read_feature_line(line, molecules_retained=nothing)
    cols = split(line, "\t")
    if length(cols) != 9
        error("feature line has wrong number of columns: $line")
    end
    str_elements, str_charges, str_aromatics, str_nbondeds, str_bonds, str_angles,
    str_propers, str_impropers, str_molecule_inds = cols

    elements = parse.(Int, split(str_elements, ","))
    # Formal charge spread over identical atoms so can be non-integer
    formal_charges = parse.(T, split(str_charges, ","))
    aromatics = parse.(T, split(str_aromatics, ","))
    nbondeds = parse.(Int, split(str_nbondeds, ","))
    molecule_inds = parse.(Int, split(str_molecule_inds, ","))

    if !isnothing(molecules_retained)
        atoms_retained = in.(molecule_inds, (molecules_retained,))
        n_atoms = sum(atoms_retained)
        elements = elements[atoms_retained]
        formal_charges = formal_charges[atoms_retained]
        aromatics = aromatics[atoms_retained]
        nbondeds = nbondeds[atoms_retained]
        molecule_inds = molecule_inds[atoms_retained]
    else
        n_atoms = length(elements)
    end

    bond_is, bond_js = Int[], Int[]
    angle_is, angle_js, angle_ks = Int[], Int[], Int[]
    proper_is, proper_js, proper_ks, proper_ls = Int[], Int[], Int[], Int[]
    improper_is, improper_js, improper_ks, improper_ls = Int[], Int[], Int[], Int[]
    adj_list = [[i] for i in 1:n_atoms]

    if str_bonds != "-"
        for str_bond in split(str_bonds, ",")
            i, j = parse.(Int, split(str_bond, "/"))
            if isnothing(molecules_retained)
                push!(bond_is, i)
                push!(bond_js, j)
                push!(adj_list[i], j)
                push!(adj_list[j], i)
            elseif all(getindex.((atoms_retained,), (i, j)))
                im, jm = sum(atoms_retained[1:i]), sum(atoms_retained[1:j])
                push!(bond_is, im)
                push!(bond_js, jm)
                push!(adj_list[im], jm)
                push!(adj_list[jm], im)
            end
        end
    end

    if str_angles != "-"
        for str_angle in split(str_angles, ",")
            i, j, k = parse.(Int, split(str_angle, "/"))
            if isnothing(molecules_retained)
                push!(angle_is, i)
                push!(angle_js, j) # Central atom
                push!(angle_ks, k)
            elseif all(getindex.((atoms_retained,), (i, j, k)))
                push!(angle_is, sum(atoms_retained[1:i]))
                push!(angle_js, sum(atoms_retained[1:j]))
                push!(angle_ks, sum(atoms_retained[1:k]))
            end
        end
    end

    if str_propers != "-" && n_proper_terms > 0
        for str_proper in split(str_propers, ",")
            i, j, k, l = parse.(Int, split(str_proper, "/"))
            if isnothing(molecules_retained)
                push!(proper_is, i)
                push!(proper_js, j)
                push!(proper_ks, k)
                push!(proper_ls, l)
            elseif all(getindex.((atoms_retained,), (i, j, k, l)))
                push!(proper_is, sum(atoms_retained[1:i]))
                push!(proper_js, sum(atoms_retained[1:j]))
                push!(proper_ks, sum(atoms_retained[1:k]))
                push!(proper_ls, sum(atoms_retained[1:l]))
            end
        end
    end

    if str_impropers != "-" && n_improper_terms > 0
        for str_improper in split(str_impropers, ",")
            i, j, k, l = parse.(Int, split(str_improper, "/"))
            if isnothing(molecules_retained)
                push!(improper_is, i) # Central atom
                push!(improper_js, j)
                push!(improper_ks, k)
                push!(improper_ls, l)
            elseif all(getindex.((atoms_retained,), (i, j, k, l)))
                push!(improper_is, sum(atoms_retained[1:i]))
                push!(improper_js, sum(atoms_retained[1:j]))
                push!(improper_ks, sum(atoms_retained[1:k]))
                push!(improper_ls, sum(atoms_retained[1:l]))
            end
        end
    end

    atom_features_in = zeros(T, n_atom_features_in, n_atoms)
    for i in 1:n_atoms
        atom_features_in[elements[i], i] = one(T)
        # Charge range is -1 to 2
        atom_features_in[18, i] = (formal_charges[i] > 0 ?  formal_charges[i] / 2 : zero(T)) # +ve charge
        atom_features_in[19, i] = (formal_charges[i] < 0 ? -formal_charges[i] / 2 : zero(T)) # -ve charge
        atom_features_in[20, i] = aromatics[i]
        atom_features_in[21 + min(nbondeds[i], 6), i] = one(T) # Can be 0 to 6
    end
    water_hs = find_water_hs(adj_list, elements)

    return elements, formal_charges, bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is,
           proper_js, proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls,
           molecule_inds, adj_list, n_atoms, atom_features_in, water_hs
end

function read_min_feature_lines(mol_ids)
    elements, formal_charges, molecule_inds = Int[], T[], Int[]
    bond_is, bond_js = Int[], Int[]
    angle_is, angle_js, angle_ks = Int[], Int[], Int[]
    proper_is, proper_js, proper_ks, proper_ls = Int[], Int[], Int[], Int[]
    improper_is, improper_js, improper_ks, improper_ls = Int[], Int[], Int[], Int[]
    adj_list = Vector{Int}[]
    atom_features_in = zeros(T, n_atom_features_in, 0)
    mol_names = String[]
    n_atoms, n_molecules = 0, 0

    for mol_id in mol_ids
        if startswith(mol_id, "protein_")
            feature_line = mol_features_prot[mol_id]
            l_mol_names = protein_data[split(mol_id, "_")[2]].mol_names
        else
            feature_line = mol_features_cond[mol_id]
            l_mol_names = split(mol_id, "_")[3:end]
        end

        l_elements, l_formal_charges, l_bond_is, l_bond_js, l_angle_is, l_angle_js, l_angle_ks,
        l_proper_is, l_proper_js, l_proper_ks, l_proper_ls, l_improper_is, l_improper_js,
        l_improper_ks, l_improper_ls, l_molecule_inds, l_adj_list, l_n_atoms,
        l_atom_features_in, _ = read_feature_line(feature_line)

        if any(startswith.(mol_id, ("vapourisation_gas_", "protein_")))
            # Vapourisation gas also provides parameters for vapourisation liquid
            n_repeats = 1
        elseif startswith(mol_id, "mixing_single_")
            n_repeats = maximum(l_molecule_inds)
        elseif startswith(mol_id, "mixing_combined_")
            n_repeats = maximum(l_molecule_inds) ÷ 2
        end
        n_atoms_rep = l_n_atoms ÷ n_repeats
        n_bonds_rep     = length(l_bond_is    ) ÷ n_repeats
        n_angles_rep    = length(l_angle_is   ) ÷ n_repeats
        n_propers_rep   = length(l_proper_is  ) ÷ n_repeats
        n_impropers_rep = length(l_improper_is) ÷ n_repeats

        append!(elements, l_elements[1:n_atoms_rep])
        append!(formal_charges, l_formal_charges[1:n_atoms_rep])
        append!(bond_is, l_bond_is[1:n_bonds_rep] .+ n_atoms)
        append!(bond_js, l_bond_js[1:n_bonds_rep] .+ n_atoms)
        append!(angle_is, l_angle_is[1:n_angles_rep] .+ n_atoms)
        append!(angle_js, l_angle_js[1:n_angles_rep] .+ n_atoms)
        append!(angle_ks, l_angle_ks[1:n_angles_rep] .+ n_atoms)
        append!(proper_is, l_proper_is[1:n_propers_rep] .+ n_atoms)
        append!(proper_js, l_proper_js[1:n_propers_rep] .+ n_atoms)
        append!(proper_ks, l_proper_ks[1:n_propers_rep] .+ n_atoms)
        append!(proper_ls, l_proper_ls[1:n_propers_rep] .+ n_atoms)
        append!(improper_is, l_improper_is[1:n_impropers_rep] .+ n_atoms)
        append!(improper_js, l_improper_js[1:n_impropers_rep] .+ n_atoms)
        append!(improper_ks, l_improper_ks[1:n_impropers_rep] .+ n_atoms)
        append!(improper_ls, l_improper_ls[1:n_impropers_rep] .+ n_atoms)
        append!(molecule_inds, l_molecule_inds[1:n_atoms_rep] .+ n_molecules)
        append!(adj_list, [al .+ n_atoms for al in l_adj_list[1:n_atoms_rep]])
        atom_features_in = hcat(atom_features_in, l_atom_features_in[:, 1:n_atoms_rep])
        append!(mol_names, l_mol_names)
        n_atoms += n_atoms_rep
        n_molecules += maximum(l_molecule_inds[1:n_atoms_rep])
    end
    @assert length(mol_names) == n_molecules
    water_hs = find_water_hs(adj_list, elements)

    return elements, formal_charges, bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is,
           proper_js, proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls,
           molecule_inds, adj_list, n_atoms, atom_features_in, water_hs, mol_names
end

function generate_neighbors(n_atoms, bond_is, bond_js, angle_is, angle_ks, proper_is,
                            proper_ls, dist_nb_cutoff)
    eligible = trues(n_atoms, n_atoms)
    for (i, j) in zip(bond_is, bond_js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(angle_is, angle_ks)
        eligible[i, k] = false
        eligible[k, i] = false
    end

    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(proper_is, proper_ls)
        special[i, l] = true
        special[l, i] = true
    end

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        special=special,
        dist_cutoff=(dist_nb_cutoff + T(0.001)),
    )

    return neighbor_finder
end

function report_parameters(atom_features, bond_features, angle_features, proper_features,
                           improper_features, partial_charges, weight_14_vdw, weight_14_coul,
                           water_hs, global_param_str)
    σs = transform_lj_σ.(atom_features[3, :])
    ϵs = transform_lj_ϵ.(atom_features[4, :], water_hs)
    strs = (
        "val-val has std charge=", std(partial_charges),
        ", med charge="    , median(partial_charges),
        ", med LJ σ="      , median(σs),
        ", std LJ σ="      , std(σs),
        ", med LJ ϵ="      , median(ϵs),
        ", std LJ ϵ="      , std(ϵs),
        ", med bond k="    , median(transform_bond_k.(bond_features[1, :], bond_features[2, :])),
        ", med bond r0="   , median(transform_bond_r0.(bond_features[1, :], bond_features[2, :])),
        ", med angle k="   , median(transform_angle_k.(angle_features[1, :], angle_features[2, :])),
        ", med angle θ0="  , median(transform_angle_θ0.(angle_features[1, :], angle_features[2, :])),
        ", std proper k="  , std(proper_features),
        ", std improper k=", std(improper_features),
        ", vdW w14="       , weight_14_vdw,
        ", Coul w14="      , weight_14_coul,
        global_param_str,
    )
    return join(string.(strs))
end

function store_string(store_id, str)
    if !isnothing(out_dir)
        open(joinpath(out_dir, "store_$store_id.txt"), "w") do of
            println(of, str)
        end
    end
end

function setup_torsions(features_pad, periodicities, phases, proper)
    return broadcast(1:size(features_pad, 2)) do i
        PeriodicTorsion{6, T, T}(
            periodicities,                      # periodicities
            phases,                             # phases
            ntuple(j -> features_pad[j, i], 6), # ks
            proper,                             # proper
        )
    end
end

# Required for performance
function ChainRulesCore.rrule(::typeof(setup_torsions), features_pad, periodicities, phases, proper)
    Y = setup_torsions(features_pad, periodicities, phases, proper)
    function setup_torsions_pullback(d_torsions)
        d_features_pad = zero(features_pad)
        for (i, d_torsion) in enumerate(d_torsions)
            d_features_pad[:, i] .= d_torsion.ks
        end
        return NoTangent(), d_features_pad, NoTangent(), NoTangent(), NoTangent()
    end
    return Y, setup_torsions_pullback
end

function multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                  molecule_inds, n_molecules)
    mol_charge_factors = zeros(T, n_molecules)
    multi_mol_charge_factors!(mol_charge_factors, charge_e_inv_s, charge_inv_s, formal_charges,
                              molecule_inds, n_molecules)
    return mol_charge_factors
end

function multi_mol_charge_factors!(mol_charge_factors, charge_e_inv_s, charge_inv_s,
                                   formal_charges, molecule_inds, n_molecules)
    for mi in 1:n_molecules
        mol_charge, mol_charge_e_inv_s, mol_charge_inv_s = zero(T), zero(T), zero(T)
        for (ai, ami) in enumerate(molecule_inds)
            if ami == mi
                mol_charge += formal_charges[ai]
                mol_charge_e_inv_s += charge_e_inv_s[ai]
                mol_charge_inv_s += charge_inv_s[ai]
            end
        end
        mol_charge_factors[mi] = (mol_charge + mol_charge_e_inv_s) / mol_charge_inv_s
    end
    return mol_charge_factors
end

function ChainRulesCore.rrule(::typeof(multi_mol_charge_factors), charge_e_inv_s, charge_inv_s,
                              formal_charges, molecule_inds, n_molecules)
    Y = multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                 molecule_inds, n_molecules)
    function multi_mol_charge_factors_pullback(d_mol_charge_factors)
        mol_charge_factors = zeros(T, n_molecules)
        d_charge_e_inv_s = zero(charge_e_inv_s)
        d_charge_inv_s = zero(charge_inv_s)
        Enzyme.autodiff(
            Enzyme.Reverse,
            multi_mol_charge_factors!,
            Enzyme.Const,
            Enzyme.Duplicated(mol_charge_factors, d_mol_charge_factors),
            Enzyme.Duplicated(charge_e_inv_s, d_charge_e_inv_s),
            Enzyme.Duplicated(charge_inv_s, d_charge_inv_s),
            Enzyme.Const(formal_charges),
            Enzyme.Const(molecule_inds),
            Enzyme.Const(n_molecules),
        )
        return NoTangent(), d_charge_e_inv_s, d_charge_inv_s, NoTangent(),
               NoTangent(), NoTangent()
    end
    return Y, multi_mol_charge_factors_pullback
end

function calc_embeddings(mol_id, feature_line, adj_list, atom_features_in, n_atoms, n_repeats,
                         bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
                         proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls,
                         model_atom_embed, model_atom_features)
    if any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_")))
        n_atoms_rep = n_atoms ÷ n_repeats
        molecule_graph = GNNGraph(adj_list[1:n_atoms_rep])
        atom_embeddings = model_atom_embed(molecule_graph, atom_features_in[:, 1:n_atoms_rep])
        atom_features = repeat(model_atom_features(atom_embeddings), 1, n_repeats)
        n_bonds_rep     = length(bond_is    ) ÷ n_repeats
        n_angles_rep    = length(angle_is   ) ÷ n_repeats
        n_propers_rep   = length(proper_is  ) ÷ n_repeats
        n_impropers_rep = length(improper_is) ÷ n_repeats
        bond_emb_1     = atom_embeddings[:, bond_is[1:n_bonds_rep]]
        bond_emb_2     = atom_embeddings[:, bond_js[1:n_bonds_rep]]
        angle_emb_1    = atom_embeddings[:, angle_is[1:n_angles_rep]]
        angle_emb_2    = atom_embeddings[:, angle_js[1:n_angles_rep]]
        angle_emb_3    = atom_embeddings[:, angle_ks[1:n_angles_rep]]
        proper_emb_1   = atom_embeddings[:, proper_is[1:n_propers_rep]]
        proper_emb_2   = atom_embeddings[:, proper_js[1:n_propers_rep]]
        proper_emb_3   = atom_embeddings[:, proper_ks[1:n_propers_rep]]
        proper_emb_4   = atom_embeddings[:, proper_ls[1:n_propers_rep]]
        improper_emb_1 = atom_embeddings[:, improper_is[1:n_impropers_rep]]
        improper_emb_2 = atom_embeddings[:, improper_js[1:n_impropers_rep]]
        improper_emb_3 = atom_embeddings[:, improper_ks[1:n_impropers_rep]]
        improper_emb_4 = atom_embeddings[:, improper_ls[1:n_impropers_rep]]
    elseif startswith(mol_id, "protein_")
        mols_unique = protein_data[split_grad_safe(mol_id, "_")[2]].mols_unique
        _, _, m_bond_is, m_bond_js, m_angle_is, m_angle_js, m_angle_ks, m_proper_is, m_proper_js,
        m_proper_ks, m_proper_ls, m_improper_is, m_improper_js, m_improper_ks, m_improper_ls, _,
        m_adj_list, _, m_atom_features_in, _ = read_feature_line(feature_line, mols_unique)
        m_molecule_graph = GNNGraph(m_adj_list)
        m_atom_embeddings = model_atom_embed(m_molecule_graph, m_atom_features_in)
        atom_features = model_atom_features(m_atom_embeddings)
        bond_emb_1     = m_atom_embeddings[:, m_bond_is]
        bond_emb_2     = m_atom_embeddings[:, m_bond_js]
        angle_emb_1    = m_atom_embeddings[:, m_angle_is]
        angle_emb_2    = m_atom_embeddings[:, m_angle_js]
        angle_emb_3    = m_atom_embeddings[:, m_angle_ks]
        proper_emb_1   = m_atom_embeddings[:, m_proper_is]
        proper_emb_2   = m_atom_embeddings[:, m_proper_js]
        proper_emb_3   = m_atom_embeddings[:, m_proper_ks]
        proper_emb_4   = m_atom_embeddings[:, m_proper_ls]
        improper_emb_1 = m_atom_embeddings[:, m_improper_is]
        improper_emb_2 = m_atom_embeddings[:, m_improper_js]
        improper_emb_3 = m_atom_embeddings[:, m_improper_ks]
        improper_emb_4 = m_atom_embeddings[:, m_improper_ls]
    else
        molecule_graph = GNNGraph(adj_list)
        atom_embeddings = model_atom_embed(molecule_graph, atom_features_in)
        atom_features = model_atom_features(atom_embeddings)
        bond_emb_1     = atom_embeddings[:, bond_is]
        bond_emb_2     = atom_embeddings[:, bond_js]
        angle_emb_1    = atom_embeddings[:, angle_is]
        angle_emb_2    = atom_embeddings[:, angle_js]
        angle_emb_3    = atom_embeddings[:, angle_ks]
        proper_emb_1   = atom_embeddings[:, proper_is]
        proper_emb_2   = atom_embeddings[:, proper_js]
        proper_emb_3   = atom_embeddings[:, proper_ks]
        proper_emb_4   = atom_embeddings[:, proper_ls]
        improper_emb_1 = atom_embeddings[:, improper_is]
        improper_emb_2 = atom_embeddings[:, improper_js]
        improper_emb_3 = atom_embeddings[:, improper_ks]
        improper_emb_4 = atom_embeddings[:, improper_ls]
    end
    return atom_features, bond_emb_1, bond_emb_2, angle_emb_1, angle_emb_2, angle_emb_3,
           proper_emb_1, proper_emb_2, proper_emb_3, proper_emb_4, improper_emb_1,
           improper_emb_2, improper_emb_3, improper_emb_4
end

function mol_to_system(mol_id, feature_line, coords, boundary, model_atom_embed, model_bond_pool,
                       model_angle_pool, model_proper_pool, model_improper_pool, model_atom_features,
                       model_bond_features, model_angle_features, model_proper_features,
                       model_improper_features, model_global_params, dist_nb_cutoff=dist_nb_cutoff)
    elements, formal_charges, bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
    proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls, molecule_inds,
    adj_list, n_atoms, atom_features_in, water_hs = read_feature_line(feature_line)
    n_molecules = maximum(molecule_inds)
    # All enthalpy of mixing cases are equimolar
    n_repeats = (startswith(mol_id, "mixing_combined_") ? (n_molecules ÷ 2) : n_molecules)

    atom_features, bond_emb_1, bond_emb_2, angle_emb_1, angle_emb_2, angle_emb_3, proper_emb_1,
    proper_emb_2, proper_emb_3, proper_emb_4, improper_emb_1, improper_emb_2, improper_emb_3,
    improper_emb_4 = calc_embeddings(mol_id, feature_line, adj_list, atom_features_in, n_atoms, n_repeats,
                bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
                proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls,
                model_atom_embed, model_atom_features)

    bond_com_emb_1 = cat(bond_emb_1, bond_emb_2; dims=1)
    bond_com_emb_2 = cat(bond_emb_2, bond_emb_1; dims=1)
    bond_pool_1 = model_bond_pool(bond_com_emb_1)
    bond_pool_2 = model_bond_pool(bond_com_emb_2)
    bond_embeddings = bond_pool_1 .+ bond_pool_2

    angle_com_emb_1 = cat(angle_emb_1, angle_emb_2, angle_emb_3; dims=1)
    angle_com_emb_2 = cat(angle_emb_3, angle_emb_2, angle_emb_1; dims=1)
    angle_pool_1 = model_angle_pool(angle_com_emb_1)
    angle_pool_2 = model_angle_pool(angle_com_emb_2)
    angle_embeddings = angle_pool_1 .+ angle_pool_2

    proper_com_emb_1 = cat(proper_emb_1, proper_emb_2, proper_emb_3, proper_emb_4; dims=1)
    proper_com_emb_2 = cat(proper_emb_4, proper_emb_3, proper_emb_2, proper_emb_1; dims=1)
    proper_pool_1 = model_proper_pool(proper_com_emb_1)
    proper_pool_2 = model_proper_pool(proper_com_emb_2)
    proper_embeddings = proper_pool_1 .+ proper_pool_2

    # Follows the convention in Espaloma except atom i not k is the central atom
    # Later this is changed when creating the interaction
    improper_com_emb_1 = cat(improper_emb_1, improper_emb_2, improper_emb_3, improper_emb_4; dims=1)
    improper_com_emb_2 = cat(improper_emb_1, improper_emb_3, improper_emb_2, improper_emb_4; dims=1)
    improper_com_emb_3 = cat(improper_emb_1, improper_emb_4, improper_emb_2, improper_emb_3; dims=1)
    improper_pool_1 = model_improper_pool(improper_com_emb_1)
    improper_pool_2 = model_improper_pool(improper_com_emb_2)
    improper_pool_3 = model_improper_pool(improper_com_emb_3)
    improper_embeddings = improper_pool_1 .+ improper_pool_2 .+ improper_pool_3

    if any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_")))
        bond_features = repeat(model_bond_features(bond_embeddings), 1, n_repeats)
        angle_features = repeat(model_angle_features(angle_embeddings), 1, n_repeats)
        proper_features = repeat(model_proper_features(proper_embeddings), 1, n_repeats)
        improper_features = repeat(model_improper_features(improper_embeddings), 1, n_repeats)
    elseif startswith(mol_id, "protein_")
        prot_data = protein_data[split_grad_safe(mol_id, "_")[2]]
        atom_features = atom_features[:, prot_data.atom_mapping]
        bond_features = model_bond_features(bond_embeddings)[:, prot_data.bond_mapping]
        angle_features = model_angle_features(angle_embeddings)[:, prot_data.angle_mapping]
        # Assume only one unrepeated molecule has torsions
        proper_features = model_proper_features(proper_embeddings)
        improper_features = model_improper_features(improper_embeddings)
        @assert size(atom_features , 2)    == n_atoms
        @assert size(bond_features , 2)    == length(bond_is)
        @assert size(angle_features, 2)    == length(angle_is)
        @assert size(proper_features, 2)   == length(proper_is)
        @assert size(improper_features, 2) == length(improper_is)
    else
        bond_features = model_bond_features(bond_embeddings)
        angle_features = model_angle_features(angle_embeddings)
        proper_features = model_proper_features(proper_embeddings)
        improper_features = model_improper_features(improper_embeddings)
    end
    proper_features_pad = cat(proper_features, zeros(T, 6 - n_proper_terms, length(proper_is)); dims=1)
    improper_features_pad = cat(improper_features, zeros(T, 6 - n_improper_terms, length(improper_is)); dims=1)

    # See Wang et al. 2022 equation 13 and Wang et al. 2019
    charge_e = atom_features[1, :]
    charge_inv_s = inv.(atom_features[2, :])
    charge_e_inv_s = charge_e .* charge_inv_s

    if any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_single_")))
        n_atoms_mol = n_atoms ÷ n_molecules
        charge_factor = (sum(formal_charges[1:n_atoms_mol]) + sum(charge_e_inv_s[1:n_atoms_mol])) /
                        sum(charge_inv_s[1:n_atoms_mol])
        charge_factors = fill(charge_factor, n_atoms)
    elseif n_molecules == 1
        charge_factor = (sum(formal_charges) + sum(charge_e_inv_s)) / sum(charge_inv_s)
        charge_factors = fill(charge_factor, n_atoms)
    else
        mol_charge_factors = multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                                      molecule_inds, n_molecules)
        charge_factors = [mol_charge_factors[mi] for mi in molecule_inds]
    end

    partial_charges = -charge_e_inv_s .+ charge_inv_s .* charge_factors
    weight_14_vdw_raw = model_global_params()[1]
    # 1-4 weight is 0 for other functional forms since OpenMM custom force doesn't support it
    if vdw_functional_form == "lj" && mixing_function == "lb"
        weight_14_vdw = sigmoid(weight_14_vdw_raw)
    else
        weight_14_vdw = zero(T)
    end
    torsion_ks_size = zero(T)
    if length(proper_features) > 0
        torsion_ks_size += mean(abs, proper_features)
    end
    if improper_regularisation && length(improper_features) > 0
        torsion_ks_size += mean(abs, improper_features)
    end

    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
        σs = transform_lj_σ.(atom_features[3, :])
        ϵs = transform_lj_ϵ.(atom_features[4, :], water_hs)
        # Atom mass is never used
        atoms = [Atom(i, 1, one(T), partial_charges[i], σs[i], ϵs[i]) for i in 1:n_atoms]
        if vdw_functional_form == "lj"
            inter_vdw = LennardJones(DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut,
                                        σ_mixing, ϵ_mixing, weight_14_vdw)
        elseif vdw_functional_form == "lj69"
            inter_vdw = Mie(6, 9, DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut,
                                        σ_mixing, ϵ_mixing, weight_14_vdw, 1)
        elseif vdw_functional_form == "dexp"
            α = transform_dexp_α(model_global_params()[3])
            β = transform_dexp_β(model_global_params()[4])
            inter_vdw = DoubleExponential(α, β, σ_mixing, ϵ_mixing, weight_14_vdw, dist_nb_cutoff)
        elseif vdw_functional_form == "buff"
            δ = transform_buff_δ(model_global_params()[3])
            γ = transform_buff_γ(model_global_params()[4])
            inter_vdw = Buffered147(δ, γ, σ_mixing, ϵ_mixing, weight_14_vdw, dist_nb_cutoff)
        end
    elseif vdw_functional_form == "buck"
        As = transform_buck_A.(atom_features[3, :], water_hs)
        Bs = transform_buck_B.(atom_features[4, :])
        Cs = transform_buck_C.(atom_features[5, :], water_hs)
        atoms = [BuckinghamAtom(i, 1, one(T), partial_charges[i], As[i], Bs[i], Cs[i])
                 for i in 1:n_atoms]
        inter_vdw = Buckingham(weight_14_vdw, dist_nb_cutoff)
    elseif vdw_functional_form == "nn"
        atoms = [NNAtom(i, 1, one(T), partial_charges[i], atom_features[3:end, i])
                 for i in 1:n_atoms]
        inter_vdw = NNPairwise(model_global_params()[2:end], weight_14_vdw, dist_nb_cutoff)
    end
    @assert length(atoms) == length(coords)

    neighbor_finder = generate_neighbors(n_atoms, bond_is, bond_js, angle_is, angle_ks,
                                         proper_is, proper_ls, dist_nb_cutoff)
    velocities = zero(coords)

    if !use_coul || vdw_functional_form == "nn"
        pairwise_inters = (use_vdw ? (inter_vdw,) : ())
        general_inters = ()
        weight_14_coul = zero(T)
    else
        weight_14_coul_raw = model_global_params()[2]
        weight_14_coul = sigmoid(weight_14_coul_raw)
        periodic = any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_", "protein_")))
        if !periodic || nonbonded_method == "none"
            inter_coul = Coulomb(DistanceCutoff(dist_nb_cutoff), true, weight_14_coul,
                                 T(ustrip(Molly.coulomb_const)))
            pairwise_inters = (use_vdw ? (inter_vdw, inter_coul) : (inter_coul,))
            general_inters = ()
        elseif nonbonded_method == "cutoff"
            inter_coul = CoulombReactionField(dist_nb_cutoff, T(Molly.crf_solvent_dielectric),
                                              true, weight_14_coul, T(ustrip(Molly.coulomb_const)))
            pairwise_inters = (use_vdw ? (inter_vdw, inter_coul) : (inter_coul,))
            general_inters = ()
        elseif nonbonded_method == "pme"
            error_tol = T(0.0005)
            ew_α = inv(dist_nb_cutoff) * sqrt(-log(2 * error_tol))
            inter_coul = CoulombEwald(dist_nb_cutoff, error_tol, true, weight_14_coul,
                                      T(ustrip(Molly.coulomb_const)), ew_α, true)
            pairwise_inters = (use_vdw ? (inter_vdw, inter_coul) : (inter_coul,))
            pme = ignore_derivatives() do
                PME(dist_nb_cutoff, atoms, boundary; eligible=neighbor_finder.eligible,
                    special=neighbor_finder.special, grad_safe=true, n_threads=1)
            end
            general_inters = (pme,)
        end
    end

    if bond_functional_form == "harmonic"
        bond_inters = HarmonicBond.(
            transform_bond_k.( bond_features[1, :], bond_features[2, :]),
            transform_bond_r0.(bond_features[1, :], bond_features[2, :]),
        )
    elseif bond_functional_form == "morse"
        bond_inters = MorseBond.(
            transform_bond_k.( bond_features[1, :], bond_features[2, :]),
            transform_morse_a.(bond_features[3, :]),
            transform_bond_r0.(bond_features[1, :], bond_features[2, :]),
        )
    end
    bonds = InteractionList2Atoms(bond_is, bond_js, bond_inters)

    if angle_functional_form == "harmonic"
        angle_inters = HarmonicAngle.(
            transform_angle_k.( angle_features[1, :], angle_features[2, :]),
            transform_angle_θ0.(angle_features[1, :], angle_features[2, :]),
        )
    elseif angle_functional_form == "ub"
        angle_inters = UreyBradley.(
            transform_angle_k.( angle_features[1, :], angle_features[2, :]),
            transform_angle_θ0.(angle_features[1, :], angle_features[2, :]),
            transform_bond_k.(  angle_features[3, :], angle_features[4, :]),
            transform_ub_r0.(   angle_features[3, :], angle_features[4, :]),
        )
    end
    angles = InteractionList3Atoms(angle_is, angle_js, angle_ks, angle_inters)

    propers_inters = setup_torsions(proper_features_pad, torsion_periodicities,
                                    torsion_phases, true)
    propers = InteractionList4Atoms(proper_is, proper_js, proper_ks, proper_ls, propers_inters)

    impropers_inters = setup_torsions(improper_features_pad, torsion_periodicities,
                                      torsion_phases, false)
    # The 2/3/1/4 order puts the central atom in the third position
    impropers = InteractionList4Atoms(improper_js, improper_ks, improper_is, improper_ls,
                                      impropers_inters)

    sis = (bonds, angles, propers, impropers)
    sis_present = (
        use_bonds     && length(bond_is    ) > 0,
        use_angles    && length(angle_is   ) > 0,
        use_proptor   && length(proper_is  ) > 0,
        use_improptor && length(improper_is) > 0,
    )
    specific_inter_lists = tuple([si for (si, sip) in zip(sis, sis_present) if sip]...)

    sys = System{3, Array, T, typeof(atoms), typeof(coords), typeof(boundary), typeof(velocities),
                 typeof([]), Nothing, typeof(pairwise_inters), typeof(specific_inter_lists),
                 typeof(general_inters), typeof(()), typeof(neighbor_finder), typeof(()),
                 typeof(NoUnits), typeof(NoUnits), T, Vector{T}, Nothing, Nothing}(
        atoms, coords, boundary, velocities, [], nothing, pairwise_inters, specific_inter_lists,
        general_inters, (), neighbor_finder, (), 1, NoUnits, NoUnits, one(T), zeros(T, n_atoms),
        nothing, nothing)

    ignore_derivatives() do
        if mol_id == "val-val" && coords[1][1] == T(0.4101424) &&
                    vdw_functional_form in ("lj", "lj69", "dexp", "buff") &&
                    bond_functional_form == "harmonic"
            if vdw_functional_form == "dexp"
                global_param_str = ", α=$(inter_vdw.α), β=$(inter_vdw.β)"
            elseif vdw_functional_form == "buff"
                global_param_str = ", δ=$(inter_vdw.δ), γ=$(inter_vdw.γ)"
            else
                global_param_str = ""
            end
            store_string("val-val", report_parameters(atom_features, bond_features, angle_features,
                                        proper_features, improper_features, partial_charges,
                                        weight_14_vdw, weight_14_coul, water_hs, global_param_str))
        end
    end

    return sys, partial_charges, torsion_ks_size, elements, molecule_inds
end

function features_to_ff_xml(out_fp::AbstractString, args...)
    open(out_fp, "w") do of
        features_to_ff_xml(of, args...)
    end
end

function features_to_ff_xml(io, mol_ids_or_feature_line, model_atom_embed, model_bond_pool,
                       model_angle_pool, model_proper_pool, model_improper_pool, model_atom_features,
                       model_bond_features, model_angle_features, model_proper_features,
                       model_improper_features, model_global_params)
    vdw_functional_form == "nn" && error("cannot generate xml file for vdw functional form nn")

    if mol_ids_or_feature_line isa AbstractString
        # Feature line
        elements, formal_charges, bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
        proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls, molecule_inds,
        adj_list, n_atoms, atom_features_in, water_hs = read_feature_line(mol_ids_or_feature_line)
        n_molecules = maximum(molecule_inds)
        mol_names = ["MOL$mi" for mi in 1:n_molecules]
    else
        # Molecule IDs
        elements, formal_charges, bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
        proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls, molecule_inds,
        adj_list, n_atoms, atom_features_in, water_hs, mol_names = read_min_feature_lines(mol_ids_or_feature_line)
        n_molecules = maximum(molecule_inds)
    end

    atom_features, bond_emb_1, bond_emb_2, angle_emb_1, angle_emb_2, angle_emb_3, proper_emb_1,
    proper_emb_2, proper_emb_3, proper_emb_4, improper_emb_1, improper_emb_2, improper_emb_3,
    improper_emb_4 = calc_embeddings("none", "none", adj_list, atom_features_in, n_atoms, 1,
                bond_is, bond_js, angle_is, angle_js, angle_ks, proper_is, proper_js,
                proper_ks, proper_ls, improper_is, improper_js, improper_ks, improper_ls,
                model_atom_embed, model_atom_features)

    bond_com_emb_1 = cat(bond_emb_1, bond_emb_2; dims=1)
    bond_com_emb_2 = cat(bond_emb_2, bond_emb_1; dims=1)
    bond_pool_1 = model_bond_pool(bond_com_emb_1)
    bond_pool_2 = model_bond_pool(bond_com_emb_2)
    bond_embeddings = bond_pool_1 .+ bond_pool_2

    angle_com_emb_1 = cat(angle_emb_1, angle_emb_2, angle_emb_3; dims=1)
    angle_com_emb_2 = cat(angle_emb_3, angle_emb_2, angle_emb_1; dims=1)
    angle_pool_1 = model_angle_pool(angle_com_emb_1)
    angle_pool_2 = model_angle_pool(angle_com_emb_2)
    angle_embeddings = angle_pool_1 .+ angle_pool_2

    proper_com_emb_1 = cat(proper_emb_1, proper_emb_2, proper_emb_3, proper_emb_4; dims=1)
    proper_com_emb_2 = cat(proper_emb_4, proper_emb_3, proper_emb_2, proper_emb_1; dims=1)
    proper_pool_1 = model_proper_pool(proper_com_emb_1)
    proper_pool_2 = model_proper_pool(proper_com_emb_2)
    proper_embeddings = proper_pool_1 .+ proper_pool_2

    improper_com_emb_1 = cat(improper_emb_1, improper_emb_2, improper_emb_3, improper_emb_4; dims=1)
    improper_com_emb_2 = cat(improper_emb_1, improper_emb_3, improper_emb_2, improper_emb_4; dims=1)
    improper_com_emb_3 = cat(improper_emb_1, improper_emb_4, improper_emb_2, improper_emb_3; dims=1)
    improper_pool_1 = model_improper_pool(improper_com_emb_1)
    improper_pool_2 = model_improper_pool(improper_com_emb_2)
    improper_pool_3 = model_improper_pool(improper_com_emb_3)
    improper_embeddings = improper_pool_1 .+ improper_pool_2 .+ improper_pool_3

    bond_features = model_bond_features(bond_embeddings)
    angle_features = model_angle_features(angle_embeddings)
    proper_features = model_proper_features(proper_embeddings)
    improper_features = model_improper_features(improper_embeddings)

    charge_e = atom_features[1, :]
    charge_inv_s = inv.(atom_features[2, :])
    charge_e_inv_s = charge_e .* charge_inv_s

    mol_charge_factors = multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                                  molecule_inds, n_molecules)
    charge_factors = [mol_charge_factors[mi] for mi in molecule_inds]

    partial_charges = -charge_e_inv_s .+ charge_inv_s .* charge_factors
    weight_14_vdw_raw = model_global_params()[1]
    # 1-4 weight is 0 for other functional forms since OpenMM custom force doesn't support it
    if vdw_functional_form == "lj" && mixing_function == "lb"
        weight_14_vdw = sigmoid(weight_14_vdw_raw)
    else
        weight_14_vdw = zero(T)
    end
    weight_14_coul_raw = model_global_params()[2]
    weight_14_coul = sigmoid(weight_14_coul_raw)

    element_counts = zeros(Int, length(element_i_to_name))
    atom_types = String[]
    for ei in elements
        element_counts[ei] += 1
        el = element_i_to_name[ei]
        push!(atom_types, "$el$(element_counts[ei])")
    end
    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
        σs = transform_lj_σ.(atom_features[3, :])
        ϵs = transform_lj_ϵ.(atom_features[4, :], water_hs)
        atom_params = zip(σs, ϵs)
    elseif vdw_functional_form == "buck"
        As = transform_buck_A.(atom_features[3, :], water_hs)
        Bs = transform_buck_B.(atom_features[4, :])
        Cs = transform_buck_C.(atom_features[5, :], water_hs)
        atom_params = zip(As, Bs, Cs)
    end

    println(io, "<ForceField>")
    println(io, "  <AtomTypes>")
    for (at, ei) in zip(atom_types, elements)
        el = element_i_to_name[ei]
        println(io, "    <Type element=\"$el\" name=\"$at\" class=\"$at\" mass=\"$(atomic_masses[ei])\"/>")
    end
    println(io, "  </AtomTypes>")

    println(io, "  <Residues>")
    for mol_i in 1:n_molecules
        mol_name = mol_names[mol_i]
        # Don't write the same molecule multiple times
        mol_i == findfirst(isequal(mol_name), mol_names) || continue
        atom_inds = findall(isequal(mol_i), molecule_inds)
        set_atom_inds = Set(atom_inds)
        println(io, "    <Residue name=\"$mol_name\">")
        for ai in atom_inds
            at, pc = atom_types[ai], partial_charges[ai]
            println(io, "      <Atom name=\"$at\" type=\"$at\" charge=\"$pc\"/>")
        end
        for (bi, bj) in zip(bond_is, bond_js)
            if bi in set_atom_inds && bj in set_atom_inds
                at1, at2 = atom_types[bi], atom_types[bj]
                println(io, "      <Bond atomName1=\"$at1\" atomName2=\"$at2\"/>")
            elseif bi in set_atom_inds || bj in set_atom_inds
                error("atoms $bi and $bj are bonded but are in different molecules")
            end
        end
        println(io, "    </Residue>")
    end
    println(io, "  </Residues>")

    if length(bond_is) > 0
        if bond_functional_form == "harmonic"
            println(io, "  <HarmonicBondForce>")
            bond_ks  = transform_bond_k.( bond_features[1, :], bond_features[2, :])
            bond_r0s = transform_bond_r0.(bond_features[1, :], bond_features[2, :])
            for (ai, aj, k, r0) in zip(bond_is, bond_js, bond_ks, bond_r0s)
                at1, at2 = atom_types[ai], atom_types[aj]
                println(io, "    <Bond type1=\"$at1\" type2=\"$at2\" length=\"$r0\" k=\"$k\"/>")
            end
            println(io, "  </HarmonicBondForce>")
        elseif bond_functional_form == "morse"
            println(io, "  <CustomBondForce energy=\"D*(1-exp(-a*(r-r0)))^2\">")
            println(io, "    <PerBondParameter name=\"D\"/>")
            println(io, "    <PerBondParameter name=\"a\"/>")
            println(io, "    <PerBondParameter name=\"r0\"/>")
            bond_Ds  = transform_bond_k.( bond_features[1, :], bond_features[2, :])
            bond_as  = transform_morse_a.(bond_features[3, :])
            bond_r0s = transform_bond_r0.(bond_features[1, :], bond_features[2, :])
            for (ai, aj, D, a, r0) in zip(bond_is, bond_js, bond_Ds, bond_as, bond_r0s)
                at1, at2 = atom_types[ai], atom_types[aj]
                println(io, "    <Bond type1=\"$at1\" type2=\"$at2\" D=\"$D\" a=\"$a\" r0=\"$r0\"/>")
            end
            println(io, "  </CustomBondForce>")
        end
    end

    if length(angle_is) > 0
        println(io, "  <HarmonicAngleForce>")
        angle_fcs = transform_angle_k.( angle_features[1, :], angle_features[2, :])
        angle_θ0s = transform_angle_θ0.(angle_features[1, :], angle_features[2, :])
        for (ai, aj, ak, k, θ0) in zip(angle_is, angle_js, angle_ks, angle_fcs, angle_θ0s)
            at1, at2, at3 = atom_types[ai], atom_types[aj], atom_types[ak]
            println(io, "    <Angle type1=\"$at1\" type2=\"$at2\" type3=\"$at3\" angle=\"$θ0\" k=\"$k\"/>")
        end
        println(io, "  </HarmonicAngleForce>")
        if angle_functional_form == "ub"
            println(io, "  <AmoebaUreyBradleyForce>")
            angle_bfcs = transform_bond_k.(angle_features[3, :], angle_features[4, :])
            angle_br0s = transform_ub_r0.( angle_features[3, :], angle_features[4, :])
            for (ai, aj, ak, k, r0) in zip(angle_is, angle_js, angle_ks, angle_bfcs, angle_br0s)
                at1, at2, at3 = atom_types[ai], atom_types[aj], atom_types[ak]
                println(io, "    <UreyBradley type1=\"$at1\" type2=\"$at2\" type3=\"$at3\" k=\"$k\" d=\"$r0\"/>")
            end
            println(io, "  </AmoebaUreyBradleyForce>")
        end
    end

    if length(proper_is) > 0 || length(improper_is) > 0
        println(io, "  <PeriodicTorsionForce ordering=\"default\">")
        for (prop_i, (ai, aj, ak, al)) in enumerate(zip(proper_is, proper_js, proper_ks, proper_ls))
            at1, at2, at3, at4 = atom_types[ai], atom_types[aj], atom_types[ak], atom_types[al]
            s = "    <Proper type1=\"$at1\" type2=\"$at2\" type3=\"$at3\" type4=\"$at4\""
            for ti in 1:n_proper_terms
                tpe, tph = torsion_periodicities[ti], torsion_phases[ti]
                k = proper_features[ti, prop_i]
                s *= " periodicity$ti=\"$tpe\" phase$ti=\"$tph\" k$ti=\"$k\""
            end
            println(io, s * "/>")
        end

        for (improp_i, (ai, aj, ak, al)) in enumerate(zip(improper_is, improper_js, improper_ks, improper_ls))
            at1, at2, at3, at4 = atom_types[ai], atom_types[aj], atom_types[ak], atom_types[al]
            s = "    <Improper type1=\"$at1\" type2=\"$at2\" type3=\"$at3\" type4=\"$at4\""
            for ti in 1:n_improper_terms
                tpe, tph = torsion_periodicities[ti], torsion_phases[ti]
                k = improper_features[ti, improp_i]
                s *= " periodicity$ti=\"$tpe\" phase$ti=\"$tph\" k$ti=\"$k\""
            end
            println(io, s * "/>")
        end
        println(io, "  </PeriodicTorsionForce>")
    end

    if vdw_functional_form == "lj" && mixing_function == "lb"
        println(io, "  <NonbondedForce coulomb14scale=\"$weight_14_coul\" lj14scale=\"$weight_14_vdw\">")
        println(io, "    <UseAttributeFromResidue name=\"charge\"/>")
        for (at, (σ, ϵ)) in zip(atom_types, atom_params)
            println(io, "    <Atom type=\"$at\" sigma=\"$σ\" epsilon=\"$ϵ\"/>")
        end
        println(io, "  </NonbondedForce>")
    else
        println(io, "  <NonbondedForce coulomb14scale=\"$weight_14_coul\" lj14scale=\"0.0\">")
        println(io, "    <UseAttributeFromResidue name=\"charge\"/>")
        for at in atom_types
            println(io, "    <Atom type=\"$at\" sigma=\"1\" epsilon=\"0\"/>")
        end
        println(io, "  </NonbondedForce>")

        # The 1-4 weighting is 0 for CustomNonbondedForce with bondCutoff=3
        if vdw_functional_form == "lj"
            # All mixing rule options zero water hydrogens appropriately
            println(io, "  <CustomNonbondedForce energy=\"4*$ϵ_string*(($σ_string / r)^12 - ($σ_string / r)^6)\" bondCutoff=\"3\">")
            println(io, "    <PerParticleParameter name=\"sigma\"/>")
            println(io, "    <PerParticleParameter name=\"epsilon\"/>")
            for (at, (σ, ϵ)) in zip(atom_types, atom_params)
                println(io, "    <Atom type=\"$at\" sigma=\"$σ\" epsilon=\"$ϵ\"/>")
            end
            println(io, "  </CustomNonbondedForce>")
        elseif vdw_functional_form == "lj69"
            println(io, "  <CustomNonbondedForce energy=\"$ϵ_string*(($σ_string / r)^9 - ($σ_string / r)^6)\" bondCutoff=\"3\">")
            println(io, "    <PerParticleParameter name=\"sigma\"/>")
            println(io, "    <PerParticleParameter name=\"epsilon\"/>")
            for (at, (σ, ϵ)) in zip(atom_types, atom_params)
                println(io, "    <Atom type=\"$at\" sigma=\"$σ\" epsilon=\"$ϵ\"/>")
            end
            println(io, "  </CustomNonbondedForce>")
        elseif vdw_functional_form == "dexp"
            α = transform_dexp_α(model_global_params()[3])
            β = transform_dexp_β(model_global_params()[4])
            println(io, "  <CustomNonbondedForce energy=\"$ϵ_string*(((β*exp(α))/(α-β))*exp(-α*(r/((2^(1/6))*$σ_string)))-((α*exp(β))/(α-β))*exp(-β*(r/((2^(1/6))*$σ_string))))\" bondCutoff=\"3\">")
            println(io, "    <GlobalParameter name=\"α\" defaultValue=\"$α\"/>")
            println(io, "    <GlobalParameter name=\"β\" defaultValue=\"$β\"/>")
            println(io, "    <PerParticleParameter name=\"sigma\"/>")
            println(io, "    <PerParticleParameter name=\"epsilon\"/>")
            for (at, (σ, ϵ)) in zip(atom_types, atom_params)
                println(io, "    <Atom type=\"$at\" sigma=\"$σ\" epsilon=\"$ϵ\"/>")
            end
            println(io, "  </CustomNonbondedForce>")
        elseif vdw_functional_form == "buff"
            δ = transform_buff_δ(model_global_params()[3])
            γ = transform_buff_γ(model_global_params()[4])
            println(io, "  <CustomNonbondedForce energy=\"$ϵ_string*(((1+δ)/((r/($σ_string*(2^(1/6))))+δ))^7)*(((1+γ)/(((r/($σ_string*(2^(1/6))))^7)+γ))-2)\" bondCutoff=\"3\">")
            println(io, "    <GlobalParameter name=\"δ\" defaultValue=\"$δ\"/>")
            println(io, "    <GlobalParameter name=\"γ\" defaultValue=\"$γ\"/>")
            println(io, "    <PerParticleParameter name=\"sigma\"/>")
            println(io, "    <PerParticleParameter name=\"epsilon\"/>")
            for (at, (σ, ϵ)) in zip(atom_types, atom_params)
                println(io, "    <Atom type=\"$at\" sigma=\"$σ\" epsilon=\"$ϵ\"/>")
            end
            println(io, "  </CustomNonbondedForce>")
        elseif vdw_functional_form == "buck"
            println(io, "  <CustomNonbondedForce energy=\"W1*W2*(0.1*(0.2/r)^12 + sqrt(A1*A2)*(exp((2/((1/B1)+(1/B2)))*(-r)) - (((C1+C2)/2)/r)^6))\" bondCutoff=\"3\">")
            println(io, "    <PerParticleParameter name=\"A\"/>")
            println(io, "    <PerParticleParameter name=\"B\"/>")
            println(io, "    <PerParticleParameter name=\"C\"/>")
            println(io, "    <PerParticleParameter name=\"W\"/>")
            for (at, (A, B, C)) in zip(atom_types, atom_params)
                if water_h_zero_vdw && iszero(A) && iszero(C)
                    W = 0
                else
                    W = 1
                end
                println(io, "    <Atom type=\"$at\" A=\"$A\" B=\"$B\" C=\"$C\" W=\"$W\"/>")
            end
            println(io, "  </CustomNonbondedForce>")
        end
    end
    println(io, "</ForceField>")
end

function forces_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl,
                     sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    fs_nounits = zero(coords)
    forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, pairwise_inters_nl,
                 sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    return fs_nounits
end

function forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, pairwise_inters_nl,
                      sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    Molly.pairwise_forces_loop!(fs_nounits, nothing, nothing, nothing, atoms, coords, velocities,
                                boundary, neighbors, NoUnits, length(atoms), (), pairwise_inters_nl,
                                Val(1), Val(false), 0)
    Molly.specific_forces!(fs_nounits, nothing, atoms, coords, velocities, boundary, NoUnits, (),
                           sils_2_atoms, sils_3_atoms, sils_4_atoms, Val(false), 0)
    return fs_nounits
end

duplicated_if_present(x, dx) = (length(x) > 0 ? Enzyme.Duplicated(x, dx) : Enzyme.Const(x))

function ChainRulesCore.rrule(::typeof(forces_wrap), atoms, coords, velocities, boundary,
                              pairwise_inters_nl, sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    Y = forces_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl, sils_2_atoms,
                    sils_3_atoms, sils_4_atoms, neighbors)
    function forces_wrap_pullback(d_fs_nounits)
        fs_nounits = zero(coords)
        d_atoms = zero.(atoms)
        d_coords = zero(coords)
        if vdw_functional_form == "nn"
            # Active fails here
            # Gives zero grad for weight_special, though that is set to 1 anyway
            d_pairwise_inters_nl = zero.(pairwise_inters_nl)
            pair_enz = Enzyme.Duplicated(pairwise_inters_nl, d_pairwise_inters_nl)
        elseif length(pairwise_inters_nl) > 0
            # Active required to get non-zero grads for weight_special etc.
            pair_enz = Enzyme.Active(pairwise_inters_nl)
        else
            pair_enz = Enzyme.Const(pairwise_inters_nl)
        end
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            forces_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(fs_nounits, d_fs_nounits),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            pair_enz,
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Enzyme.Const(neighbors),
        )[1]
        pair_grad = (vdw_functional_form == "nn" ? d_pairwise_inters_nl : grads[6])
        return NoTangent(), d_atoms, d_coords, NoTangent(), NoTangent(), pair_grad,
               d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms, NoTangent()
    end
    return Y, forces_wrap_pullback
end

function pe_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl,
                 sils_2_atoms, sils_3_atoms, sils_4_atoms, general_inters, neighbors)
    pe_vec = zeros(T, 1)
    pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
             sils_2_atoms, sils_3_atoms, sils_4_atoms, general_inters, neighbors)
    return pe_vec[1]
end

function pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
                  sils_2_atoms, sils_3_atoms, sils_4_atoms, general_inters, neighbors)
    pe = Molly.pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, NoUnits,
                                length(atoms), (), pairwise_inters_nl, Val(T), Val(1), 0)

    # Molly.specific_pe gave Enzyme error
    @inbounds for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
            pe += potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j],
                                   NoUnits, velocities[i], velocities[j], 0)
        end
    end

    @inbounds for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
            pe += potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i],
                                   atoms[j], atoms[k], NoUnits, velocities[i], velocities[j],
                                   velocities[k], 0)
        end
    end

    @inbounds for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)
            pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                                   atoms[i], atoms[j], atoms[k], atoms[l], NoUnits, velocities[i],
                                   velocities[j], velocities[k], velocities[l], 0)
        end
    end

    if length(general_inters) > 0
        # Must be PME
        pe += Molly.ewald_pe_forces!(nothing, nothing, general_inters[1], atoms, coords, boundary,
                                     NoUnits, NoUnits, Val(false), false; n_threads=1)
    end

    pe_vec[1] = pe
    return pe_vec
end

function ChainRulesCore.rrule(::typeof(pe_wrap), atoms, coords, velocities, boundary,
                              pairwise_inters_nl, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                              general_inters, neighbors)
    Y = pe_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl, sils_2_atoms,
                sils_3_atoms, sils_4_atoms, general_inters, neighbors)
    function pe_wrap_pullback(d_pe)
        d_atoms = zero.(atoms)
        d_coords = zero(coords)
        if vdw_functional_form == "nn"
            d_pairwise_inters_nl = zero.(pairwise_inters_nl)
            pair_enz = Enzyme.Duplicated(pairwise_inters_nl, d_pairwise_inters_nl)
        elseif length(pairwise_inters_nl) > 0
            pair_enz = Enzyme.Active(pairwise_inters_nl)
        else
            pair_enz = Enzyme.Const(pairwise_inters_nl)
        end
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        d_general_inters = zero.(general_inters)
        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            pe_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(zeros(T, 1), [d_pe]),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            pair_enz,
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            duplicated_if_present(general_inters, d_general_inters),
            Enzyme.Const(neighbors),
        )[1]
        pair_grad = (vdw_functional_form == "nn" ? d_pairwise_inters_nl : grads[6])
        return NoTangent(), d_atoms, d_coords, NoTangent(), NoTangent(), pair_grad,
               d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms, d_general_inters, NoTangent()
    end
    return Y, pe_wrap_pullback
end

function mol_to_preds(mol_id, args...)
    sys, partial_charges, torsion_ks_size, elements, molecule_inds = mol_to_system(mol_id, args...)
    neighbors = find_neighbors(sys; n_threads=1)
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
    if any(startswith.(mol_id, ("vapourisation_", "mixing_", "protein_")))
        fs = nothing
        pe = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                        sils_2_atoms, sils_3_atoms, sils_4_atoms, sys.general_inters, neighbors)
    else
        # Assume no general interactions for QM systems
        fs = forces_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
        pe = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                        sils_2_atoms, sils_3_atoms, sils_4_atoms, sys.general_inters, neighbors)
    end
    return fs, pe, partial_charges, torsion_ks_size, elements, molecule_inds
end

function read_coordinates(mol_hdf5::Union{HDF5.Group, Dict}, conf_i)
    conformation = mol_hdf5["conformations"][:, :, conf_i]::Matrix{T}
    return SVector{3, T}.(eachcol(conformation .* bohr_to_nm))
end

function read_dft_forces(mol_hdf5::Union{HDF5.Group, Dict}, conf_i)
    if startswith(mol_hdf5["subset"][1], "RNA")
        dft_total_gradient = mol_hdf5["dft_total_gradient"][:, :, conf_i]::Matrix{T} .+
                             mol_hdf5["dispersion_correction_gradient"][:, :, conf_i]::Matrix{T}
    else
        dft_total_gradient = mol_hdf5["dft_total_gradient"][:, :, conf_i]::Matrix{T}
    end
    dft_fs = SVector{3, T}.(eachcol(-dft_total_gradient .* force_conversion))
    exceeds_max = maximum(norm.(dft_fs)) > max_force_norm
    return dft_fs, exceeds_max
end

function read_dft_pe(mol_hdf5::Union{HDF5.Group, Dict}, conf_i)
    if startswith(mol_hdf5["subset"][1], "RNA")
        dft_pe_hartree = T(mol_hdf5["dft_total_energy"][conf_i]) +
                         T(mol_hdf5["dispersion_correction_energy"][conf_i])
    else
        dft_pe_hartree = T(mol_hdf5["dft_total_energy"][conf_i])
    end
    return dft_pe_hartree * hartree_to_kJpmol
end

function read_dft_charges(mol_hdf5::Union{HDF5.Group, Dict}, conf_i)
    if skip_water_charges && mol_hdf5["subset"][1] == "SPICE Water Clusters v1.0"
        return T[], false
    elseif haskey(mol_hdf5, "mbis_charges") # Missing for the RNA data and 121 SPICE molecules
        dft_charges = mol_hdf5["mbis_charges"][1, :, conf_i]::Vector{T}
        has_charges = true
        return dft_charge_weight .* dft_charges, has_charges
    else
        return T[], false
    end
end

function read_coordinates(xyz::XYZFile, conf_i)
    coords_Å = map(readlines(xyz.filepath)[3:end]) do line
        SVector{3, T}(parse.(T, split(line)[2:4]))
    end
    return coords_Å ./ 10 # Convert to nm
end

function read_dft_forces(xyz::XYZFile, conf_i)
    dft_fs_ev_per_Å = map(readlines(xyz.filepath)[3:end]) do line
        SVector{3, T}(parse.(T, split(line)[5:7]))
    end
    dft_fs = dft_fs_ev_per_Å .* eVpÅ_to_kJpmolpnm
    exceeds_max = maximum(norm.(dft_fs)) > max_force_norm
    return dft_fs, exceeds_max
end

function read_dft_pe(xyz::XYZFile, conf_i)
    return zero(T) # Only one conformation for MACE-OFF23 data so no energy comparison
end

function read_dft_charges(xyz::XYZFile, conf_i)
    return T[], false # Missing for MACE-OFF23 data
end

function read_gems_array(col_name, conf_i)
    query_str = "SELECT $col_name FROM data WHERE id=$(conf_i-1)" # Zero-based indexing
    df = DataFrame(DBInterface.execute(gems_db, query_str))
    arr_uint8 = only(df[!, col_name])
    arr_f32 = map(1:4:length(arr_uint8)) do i
        read(IOBuffer(arr_uint8[i:(i + 3)]), Float32)
    end
    return reinterpret(SVector{3, T}, T.(arr_f32))
end

function read_coordinates_gems(conf_i)
    coords_Å = read_gems_array("R", conf_i)
    return coords_Å ./ 10 # Convert to nm
end

function read_dft_forces_gems(conf_i)
    fs_ev_per_Å = read_gems_array("F", conf_i)
    return fs_ev_per_Å .* eVpÅ_to_kJpmolpnm
end

function read_sim_data(mol_id, training_sim_dir, frame_i, temp=nothing)
    exp_type, sim_type, smiles = split(mol_id, "_"; limit=3)
    if exp_type == "vapourisation"
        traj_fp = "$(smiles)_$(Int(temp))K.dcd"
    else
        traj_fp = "$smiles.dcd"
    end
    traj = Chemfiles.Trajectory(joinpath(training_sim_dir, "$(exp_type)_$sim_type", traj_fp))
    frame = Chemfiles.read_step(traj, frame_i - 1) # Zero-based indexing
    pos = Chemfiles.positions(frame)
    coords_unordered = SVector{3, T}.(eachcol(pos)) ./ 10 # Convert to nm
    if exp_type == "mixing" && sim_type == "combined"
        # PDB files have molecules in order 1,1,2,2 so we reorder to 1,2,1,2
        molecule_inds_str = split(mol_features_cond[mol_id], "\t")[end]
        molecule_inds = parse.(Int, split(molecule_inds_str, ","))
        n_molecules_each = maximum(molecule_inds) ÷ 2
        n_atoms_1 = findlast(isequal(1), molecule_inds)
        n_atoms_2 = findlast(isequal(2), molecule_inds) - n_atoms_1
        n_atoms_com = n_atoms_1 + n_atoms_2
        n_atoms_1_all = n_atoms_1 * n_molecules_each
        coords = zero(coords_unordered)
        for mi in 1:n_molecules_each
            for ai in 1:n_atoms_1
                coords[(mi-1) * n_atoms_com + ai] = coords_unordered[(mi-1) * n_atoms_1 + ai]
            end
            for ai in 1:n_atoms_2
                coords[(mi-1) * n_atoms_com + n_atoms_1 + ai] = coords_unordered[n_atoms_1_all + (mi-1) * n_atoms_2 + ai]
            end
        end
    else
        coords = coords_unordered
    end
    if sim_type == "gas"
        boundary = boundary_inf
    else
        box_sides = T.(Chemfiles.lengths(Chemfiles.UnitCell(frame))) ./ 10 # Convert to nm
        boundary = CubicBoundary(box_sides...)
        coords .= wrap_coords.(coords, (boundary,))
    end
    return coords, boundary
end

# See Kovacs2025 and Magdau2023
function split_forces(fs, coords, molecule_inds, element_inds)
    forces_intra, forces_inter = zero(fs), zero(fs)
    split_forces!(forces_intra, forces_inter, fs, coords, molecule_inds, element_inds)
    return forces_intra, forces_inter
end

function split_forces!(forces_intra, forces_inter, fs, coords, molecule_inds, element_inds)
    n_molecules = maximum(molecule_inds)
    if n_molecules == 1
        forces_inter .= (zero(SVector{3, T}),)
        forces_intra .= fs
    else
        mol_f_trans = zeros(SVector{3, T}, n_molecules)
        mol_masses = zeros(n_molecules)
        for (f, mi, el) in zip(fs, molecule_inds, element_inds)
            mol_f_trans[mi] += f
            mol_masses[mi] += atomic_masses[el]
        end

        forces_trans = zero(fs)
        mol_coms = zeros(SVector{3, T}, n_molecules)
        for (ai, (mi, el, coord)) in enumerate(zip(molecule_inds, element_inds, coords))
            atom_mass_frac = atomic_masses[el] / mol_masses[mi]
            forces_trans[ai] = atom_mass_frac * mol_f_trans[mi]
            mol_coms[mi] += atom_mass_frac * coord
        end

        mol_torques = zeros(SVector{3, T}, n_molecules)
        for (f, mi, coord) in zip(fs, molecule_inds, coords)
            mol_torques[mi] += f × (coord - mol_coms[mi])
        end

        mol_Is = zeros(n_molecules, 3, 3)
        for (mi, el, coord) in zip(molecule_inds, element_inds, coords)
            atom_mass = atomic_masses[el]
            coord_com = coord - mol_coms[mi]
            sqdist_com = sum(abs2, coord_com)
            for i in 1:3, j in 1:3
                if i == j
                    mol_Is[mi, i, j] += atom_mass * (sqdist_com - coord_com[i]^2)
                else
                    mol_Is[mi, i, j] += atom_mass * (-coord_com[i] * coord_com[j])
                end
            end
        end

        mol_inv_Is = zeros(SMatrix{3, 3, T}, n_molecules)
        for mi in 1:n_molecules
            if all(iszero, mol_Is[mi, :, :])
                mol_inv_Is[mi] = SMatrix{3, 3, T}(I)
            else
                mol_inv_Is[mi] = inv(SMatrix{3, 3, T}(mol_Is[mi, :, :]))
            end
        end

        forces_rot = zero(fs)
        for (ai, (mi, el, coord)) in enumerate(zip(molecule_inds, element_inds, coords))
            I_T_mvp = mol_inv_Is[mi] * mol_torques[mi]
            forces_rot[ai] = atomic_masses[el] * (coord - mol_coms[mi]) × I_T_mvp
        end

        forces_inter .= forces_trans .+ forces_rot
        forces_intra .= fs .- forces_inter
    end
    return forces_intra, forces_inter
end

function ChainRulesCore.rrule(::typeof(split_forces), fs, coords, molecule_inds, element_inds)
    Y = split_forces(fs, coords, molecule_inds, element_inds)
    function split_forces_pullback(d_fs_both)
        d_coords = zero(coords)
        d_fs = zero(fs)
        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            split_forces!,
            Enzyme.Const,
            Enzyme.Duplicated(zero(fs), d_fs_both[1]),
            Enzyme.Duplicated(zero(fs), d_fs_both[2]),
            Enzyme.Duplicated(fs, d_fs),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(molecule_inds),
            Enzyme.Const(element_inds),
        )[1]
        return NoTangent(), d_fs, d_coords, NoTangent(), NoTangent()
    end
    return Y, split_forces_pullback
end

abs2_vec(x) = abs2.(x)
force_loss(    fs, dft_fs) = loss_weight_force * mean(sqrt.(sum.(abs2_vec.(fs .- dft_fs))))
force_loss_mse(fs, dft_fs) = loss_weight_force * mean(      sum.(abs2_vec.(fs .- dft_fs)))
pe_loss(pe_diff, dft_pe_diff) = loss_weight_energy * abs(pe_diff - dft_pe_diff)
pe_loss_mse(pe_diff, dft_pe_diff) = loss_weight_energy * (pe_diff - dft_pe_diff) ^ 2
torsion_ks_loss(torsion_ks_size) = loss_weight_torsion_ks * torsion_ks_size

function charge_loss(charges, dft_charges)
    if charge_sq_loss
        return loss_weight_charge * mean(abs2.(charges .- dft_charges))
    else
        return loss_weight_charge * mean(abs.(charges .- dft_charges))
    end
end

charge_regularisation(partial_charges) = loss_weight_charge_reg * mean(abs.(partial_charges))

function param_regularisation(models)
    s = sum(abs2, Flux.destructure(models[1:(end-1)])[1])
    # Global parameters excluded from regularisation except for NNPairwise NN params
    if vdw_functional_form == "nn"
        s += sum(abs2, Flux.destructure(models[end])[1][2:end])
    end
    return loss_weight_regularisation * s
end

calc_RT(temp) = ustrip(u"kJ/mol", T(Unitful.R) * temp * u"K")

function enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    # See https://docs.openforcefield.org/projects/evaluator/en/stable/properties/properties.html
    RT = calc_RT(temp)
    ΔH_vap = mean_U_gas - snapshot_U_liquid / n_molecules + RT
    return ΔH_vap
end

function enth_vap_loss(snapshot_U_liquid, mean_U_gas, temp, frame_i, repeat_i, n_molecules, mol_id)
    ΔH_vap = enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    ΔH_vap_exp = T(enth_vap_exp_data[mol_id](temp))
    loss_ΔH_vap = loss_weight_enth_vap * abs(ΔH_vap - ΔH_vap_exp)
    ignore_derivatives() do
        if mol_id == "vapourisation_liquid_O" && temp == T(295.0) && frame_i == 101 && repeat_i == 1
            store_string("ΔHvap", "ΔHvap water $ΔH_vap, exp $ΔH_vap_exp, loss $loss_ΔH_vap")
        end
    end
    return loss_ΔH_vap
end

const pressure_enth_mixing = ustrip(T, u"kJ * mol^-1", 1.0u"bar" * 1.0u"nm^3" * Unitful.Na)

function enth_mixing_loss(pe_com, pe_1, pe_2, bound_com, bound_1, bound_2,
                          n_mols_com, n_mols_1, n_mols_2, mol_id, frame_i, repeat_i)
    u_mol_com = (pe_com + pressure_enth_mixing * Molly.volume(bound_com)) / n_mols_com
    u_mol_1   = (pe_1   + pressure_enth_mixing * Molly.volume(bound_1)  ) / n_mols_1
    u_mol_2   = (pe_2   + pressure_enth_mixing * Molly.volume(bound_2)  ) / n_mols_2
    ΔH_mix = u_mol_com - (u_mol_1 + u_mol_2) / 2 # β cancels out
    ΔH_mix_exp = enth_mixing_exp_data[mol_id]
    loss_ΔH_mix = loss_weight_enth_mixing * abs(ΔH_mix - ΔH_mix_exp)
    ignore_derivatives() do
        if mol_id == "mixing_combined_CCCCO_OC1=NCCC1" && frame_i == 101 && repeat_i == 1
            store_string("ΔHmix", "ΔHmix CCCCO_OC1=NCCC1 $ΔH_mix ($u_mol_com $u_mol_1 $u_mol_2), exp $ΔH_mix_exp, loss $loss_ΔH_mix")
        end
    end
    return loss_ΔH_mix
end

# See Cavender2025 and Takaba2024
function J_coupling_loss(protein, coords, boundary)
    A, B, C = T(7.97), T(-1.26), T(0.63)
    Δ = deg2rad(T(-60.0))
    χ2_sum = zero(T)
    χ2_count = 0
    for d in J_coupling_data[protein]
        if d.present
            θ = torsion_angle(coords[d.i], coords[d.j], coords[d.k], coords[d.l], boundary)
            cos_θpΔ = cos(θ + Δ)
            J_comp = A * cos_θpΔ^2 + B * cos_θpΔ + C
            J_exp, σ = d.J, d.σ
            χ2 = (J_comp - J_exp)^2 / σ^2
            χ2_sum += χ2
            χ2_count += 1
        end
    end
    return loss_weight_J_coupling * χ2_sum / χ2_count
end

chem_shift_loss(cs_sim, cs_exp) = loss_weight_chem_shift * mean(abs, cs_sim .- cs_exp)

multiply_grads(grads, x) = fmap(i -> (isnothing(i) ? nothing : i * x), grads)

accum_grads(x, y) = Zygote.accum(x, y)

# Fails on Dropout layer so define that special case
# This is type piracy but avoids defining all the methods
function Zygote.accum(::NamedTuple{(:p, :dims, :active, :rng), T},
                      ::NamedTuple{(:p, :dims, :active, :rng), T}) where T
    return nothing
end

Zygote.accum(::NamedTuple{(:p, :dims, :active, :rng), <:Any}, ::Nothing) = nothing
Zygote.accum(::Nothing, ::NamedTuple{(:p, :dims, :active, :rng), <:Any}) = nothing

check_no_nans(grads) = !any(g -> any(isnan, Flux.destructure(g)[1]), grads)

function pe_frame(mol_id, feature_line, coords, boundary, models...)
    sys, _, _, _, _ = mol_to_system(mol_id, feature_line, coords, boundary, models...)
    neighbors = find_neighbors(sys)
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
    pe = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary,
                 sys.pairwise_inters, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                 sys.general_inters, neighbors)
    return pe
end

# This function implements an incorrect equation compared to the equation in the paper
# It is left incorrect here since it was used to train the model
# A correct implementation may give better results
function protein_loss_grads(mol_id, training_sim_dir, traj_pes, traj_chem_shifts, frames,
                            calc_grads, models...)
    TZ = Float64 # Use higher precision for partition functions
    _, protein = split(mol_id, "_")
    feature_line = mol_features_prot["protein_$(protein)_full"]
    chem_shifts_sim, chem_shifts_exp = traj_chem_shifts[protein], chem_shift_data[protein]
    traj = Chemfiles.Trajectory(joinpath(training_sim_dir, "protein", "$protein.dcd"))

    coords_list, boundary_list = Vector{SVector{3, T}}[], CubicBoundary{3, T, T}[]
    for (i, frame_i) in enumerate(frames)
        frame = Chemfiles.read_step(traj, frame_i - 1) # Zero-based indexing
        pos = Chemfiles.positions(frame)
        coords = SVector{3, T}.(eachcol(pos)) ./ 10 # Convert to nm
        box_sides = T.(Chemfiles.lengths(Chemfiles.UnitCell(frame))) ./ 10 # Convert to nm
        boundary = CubicBoundary(box_sides...)
        coords .= wrap_coords.(coords, (boundary,))
        push!(coords_list, coords)
        push!(boundary_list, boundary)
    end
    n_atoms = length(coords_list[1])
    β = inv(calc_RT(TZ(298.0)) * n_atoms)
    pes_ref = [traj_pes[protein][frame_i] for frame_i in frames]
    Z_ref = sum(pe -> exp(-β * TZ(pe)), pes_ref)

    n_chunks = Threads.nthreads()
    loss_jc_sum_chunks, loss_cs_sum_chunks = zeros(T, n_chunks), zeros(T, n_chunks)
    Z_chunks = zeros(TZ, n_chunks)
    dE_dp_sum_chunks     = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    ljc_dE_dp_sum_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    lcs_dE_dp_sum_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    frame_count_chunks = zeros(Int, n_chunks)

    Threads.@threads for chunk_id in 1:n_chunks
        for i in chunk_id:n_chunks:length(frames)
            frame_i, coords, boundary = frames[i], coords_list[i], boundary_list[i]
            if calc_grads
                pe, grads = Zygote.withgradient(pe_frame, mol_id, feature_line, coords,
                                                boundary, models...)
                dE_dp = grads[5:end]
            else
                pe = pe_frame(mol_id, feature_line, coords, boundary, models...)
                dE_dp = convert(Vector{Any}, fill(nothing, length(models)))
            end

            if !isnan(pe) && check_no_nans(dE_dp)
                frame_count_chunks[chunk_id] += 1
                Z_chunks[chunk_id] += exp(-β * TZ(pe))
                exp_mβ_pe_diff = T(exp(-β * TZ(pe - pes_ref[i])))
                loss_jc = J_coupling_loss(protein, coords, boundary) * exp_mβ_pe_diff
                loss_cs = chem_shift_loss(chem_shifts_sim[frame_i], chem_shifts_exp) * exp_mβ_pe_diff
                loss_jc_sum_chunks[chunk_id] += loss_jc
                loss_cs_sum_chunks[chunk_id] += loss_cs
                dE_dp_sum_chunks[chunk_id] = accum_grads.(dE_dp_sum_chunks[chunk_id], dE_dp)
                ljc_dE_dp_sum_chunks[chunk_id] = accum_grads.(ljc_dE_dp_sum_chunks[chunk_id],
                                                              multiply_grads(dE_dp, loss_jc))
                lcs_dE_dp_sum_chunks[chunk_id] = accum_grads.(lcs_dE_dp_sum_chunks[chunk_id],
                                                              multiply_grads(dE_dp, loss_cs))
            end
        end
    end

    dE_dp_sum     = convert(Vector{Any}, fill(nothing, length(models)))
    ljc_dE_dp_sum = convert(Vector{Any}, fill(nothing, length(models)))
    lcs_dE_dp_sum = convert(Vector{Any}, fill(nothing, length(models)))
    for chunk_id in 1:n_chunks
        dE_dp_sum = accum_grads.(dE_dp_sum, dE_dp_sum_chunks[chunk_id])
    end
    for chunk_id in 1:n_chunks
        ljc_dE_dp_sum = accum_grads.(ljc_dE_dp_sum, ljc_dE_dp_sum_chunks[chunk_id])
        lcs_dE_dp_sum = accum_grads.(lcs_dE_dp_sum, lcs_dE_dp_sum_chunks[chunk_id])
    end

    frame_count = sum(frame_count_chunks)
    Z = sum(Z_chunks)
    loss_jc_mean = T(Z_ref / Z) * sum(loss_jc_sum_chunks) / frame_count
    loss_cs_mean = T(Z_ref / Z) * sum(loss_cs_sum_chunks) / frame_count
    minus_ljc_mean_dE_dp_mean = multiply_grads(dE_dp_sum, -loss_jc_mean / frame_count)
    minus_lcs_mean_dE_dp_mean = multiply_grads(dE_dp_sum, -loss_cs_mean / frame_count)
    ljc_dE_dp_mean = multiply_grads(ljc_dE_dp_sum, T(Z_ref / (Z * frame_count)))
    lcs_dE_dp_mean = multiply_grads(lcs_dE_dp_sum, T(Z_ref / (Z * frame_count)))
    dljc_dp = multiply_grads(accum_grads.(ljc_dE_dp_mean, minus_ljc_mean_dE_dp_mean), T(-β) * n_atoms)
    dlcs_dp = multiply_grads(accum_grads.(lcs_dE_dp_mean, minus_lcs_mean_dE_dp_mean), T(-β) * n_atoms)
    return loss_jc_mean, loss_cs_mean, dljc_dp, dlcs_dp
end

function combined_loss(mol_id, coords, dft_fs, dft_charges, models...)
    fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                    mol_id, mol_features[mol_id], coords, boundary_inf, models...)
    fs_intra, fs_inter = split_forces(fs, coords, molecule_inds, elements)
    dft_fs_intra, dft_fs_inter = split_forces(dft_fs, coords, molecule_inds, elements)
    loss_fs_intra = force_loss(fs_intra, dft_fs_intra)
    loss_fs_inter = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
    loss_pe = pe_loss(pe, T(74000.0)) # Dummy energy loss for testing
    loss_charges = charge_loss(charges, dft_charges)
    loss_charge_reg = charge_regularisation(charges)
    loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
    loss_regularisation = param_regularisation((models...,))
    return loss_fs_intra + loss_fs_inter + loss_pe + loss_charges + loss_charge_reg +
           loss_torsion_ks + loss_regularisation
end

function ChainRulesCore.rrule(TY::Type{<:Atom}, vs...)
    Y = TY(vs...)
    function Atom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.σ, Ȳ.ϵ
    end
    return Y, Atom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:BuckinghamAtom}, vs...)
    Y = TY(vs...)
    function BuckinghamAtom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.A, Ȳ.B, Ȳ.C
    end
    return Y, BuckinghamAtom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:NNAtom}, vs...)
    Y = TY(vs...)
    function NNAtom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.params
    end
    return Y, NNAtom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList2Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList2Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList2Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList3Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList3Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList3Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList4Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList4Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters,
               NoTangent()
    end
    return Y, InteractionList4Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicBond}, vs...)
    Y = TY(vs...)
    function HarmonicBond_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.r0
    end
    return Y, HarmonicBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:MorseBond}, vs...)
    Y = TY(vs...)
    function MorseBond_pullback(Ȳ)
        return NoTangent(), Ȳ.D, Ȳ.a, Ȳ.r0
    end
    return Y, MorseBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicAngle}, vs...)
    Y = TY(vs...)
    function HarmonicAngle_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.θ0
    end
    return Y, HarmonicAngle_pullback
end

function ChainRulesCore.rrule(TY::Type{<:UreyBradley}, vs...)
    Y = TY(vs...)
    function UreyBradley_pullback(Ȳ)
        return NoTangent(), Ȳ.kangle, Ȳ.θ0,  Ȳ.kbond, Ȳ.r0
    end
    return Y, UreyBradley_pullback
end

function ChainRulesCore.rrule(TY::Type{<:PeriodicTorsion}, vs...)
    Y = TY(vs...)
    function PeriodicTorsion_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.phases, Ȳ.ks, NoTangent()
    end
    return Y, PeriodicTorsion_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Coulomb}, vs...)
    Y = TY(vs...)
    function Coulomb_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.weight_special, Ȳ.coulomb_const
    end
    return Y, Coulomb_pullback
end

function ChainRulesCore.rrule(TY::Type{<:CoulombReactionField}, vs...)
    Y = TY(vs...)
    function CoulombReactionField_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.solvent_dielectric, NoTangent(),
                Ȳ.weight_special, Ȳ.coulomb_const
    end
    return Y, CoulombReactionField_pullback
end

function ChainRulesCore.rrule(TY::Type{<:CoulombEwald}, vs...)
    Y = TY(vs...)
    function CoulombEwald_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.error_tol, NoTangent(),
                Ȳ.weight_special, Ȳ.coulomb_const, Ȳ.α, NoTangent()
    end
    return Y, CoulombEwald_pullback
end

function ChainRulesCore.rrule(TY::Type{<:LennardJones}, vs...)
    Y = TY(vs...)
    function LennardJones_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), Ȳ.weight_special
    end
    return Y, LennardJones_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Mie}, vs...)
    Y = TY(vs...)
    function Mie_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Mie_pullback
end

function ChainRulesCore.rrule(TY::Type{<:DoubleExponential}, vs...)
    Y = TY(vs...)
    function DoubleExponential_pullback(Ȳ)
        return NoTangent(), Ȳ.α, Ȳ.β, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, DoubleExponential_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Buffered147}, vs...)
    Y = TY(vs...)
    function Buffered147_pullback(Ȳ)
        return NoTangent(), Ȳ.δ, Ȳ.γ, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buffered147_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Buckingham}, vs...)
    Y = TY(vs...)
    function Buckingham_pullback(Ȳ)
        return NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buckingham_pullback
end

function ChainRulesCore.rrule(TY::Type{<:NNPairwise}, vs...)
    Y = TY(vs...)
    function NNPairwise_pullback(Ȳ)
        return NoTangent(), Ȳ.params, Ȳ.weight_special, NoTangent()
    end
    return Y, NNPairwise_pullback
end

split_grad_safe(args...) = split(args...)

@non_differentiable GraphNeuralNetworks.GNNGraph(args...)
@non_differentiable Molly.find_neighbors(args...)
@non_differentiable read_feature_line(args...)
@non_differentiable generate_neighbors(args...)
@non_differentiable store_string(args...)
@non_differentiable report_parameters(args...)
@non_differentiable read_coordinates(args...)
@non_differentiable read_dft_forces(args...)
@non_differentiable read_dft_charges(args...)
@non_differentiable read_coordinates_gems(args...)
@non_differentiable read_dft_forces_gems(args...)
@non_differentiable read_sim_data(args...)
@non_differentiable calc_RT(args...)
@non_differentiable split_grad_safe(args...)

mol_id = "007 TYR"
conf_i = 1
coords = read_coordinates(h5read(spice_hdf5_fp, mol_id), conf_i)
dft_fs, exceeds_max = read_dft_forces(h5read(spice_hdf5_fp, mol_id), conf_i)
dft_pe = read_dft_pe(h5read(spice_hdf5_fp, mol_id), conf_i)
dft_charges, has_charges = read_dft_charges(h5read(spice_hdf5_fp, mol_id), conf_i)

fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(mol_id,
                                            mol_features[mol_id], coords, boundary_inf, models...)
loss = combined_loss(mol_id, coords, dft_fs, dft_charges, models...)
grads = Zygote.gradient(combined_loss, mol_id, coords, dft_fs, dft_charges, models...)
optims = [Flux.setup(optimiser(learning_rate), m) for m in models]

function torsion_pes(torsion, θs)
    return map(θs) do θ
        pe = zero(T)
        for (periodicity, phase, k) in zip(torsion.periodicities, torsion.phases, torsion.ks)
            pe += k + k * cos((periodicity * θ) - phase)
        end
        return pe
    end
end

function plot_training(plot_fp, models,
                       epochs_mean_fs_intra_train     , epochs_mean_fs_intra_val     ,
                       epochs_mean_fs_inter_train     , epochs_mean_fs_inter_val     ,
                       epochs_mean_pe_train           , epochs_mean_pe_val           ,
                       epochs_mean_charges_train      , epochs_mean_charges_val      ,
                       epochs_mean_charge_reg_train   , epochs_mean_charge_reg_val   ,
                       epochs_mean_torsion_ks_train   , epochs_mean_torsion_ks_val   ,
                       epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
                       epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems,
                       epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val     ,
                       epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val  ,
                       epochs_mean_J_coupling_train   , epochs_mean_J_coupling_val   ,
                       epochs_mean_chem_shift_train   , epochs_mean_chem_shift_val   ,
                       epochs_loss_regularisation)
    f = Figure(size=(800, 600))
    ax = Axis(f[1, 1:2],
        title="Training progress",
        xlabel="Epoch",
        ylabel="Mean loss value",
    )
    plot_fs_inter_weight = round(10 / loss_weight_force_inter; sigdigits=4)
    weighted_fs_inter_train      = epochs_mean_fs_inter_train      .* plot_fs_inter_weight
    weighted_fs_inter_val        = epochs_mean_fs_inter_val        .* plot_fs_inter_weight
    weighted_fs_inter_train_gems = epochs_mean_fs_inter_train_gems .* plot_fs_inter_weight
    weighted_fs_inter_val_gems   = epochs_mean_fs_inter_val_gems   .* plot_fs_inter_weight
    plot_pe_weight = round(1e-2 / loss_weight_energy; sigdigits=4)
    weighted_pe_train = epochs_mean_pe_train .* plot_pe_weight
    weighted_pe_val   = epochs_mean_pe_val   .* plot_pe_weight
    plot_charge_weight = round((charge_sq_loss ? 100 : 10) / loss_weight_charge; sigdigits=4)
    weighted_charges_train = epochs_mean_charges_train .* plot_charge_weight
    weighted_charges_val   = epochs_mean_charges_val   .* plot_charge_weight
    plot_charge_reg_weight = round(1.0 / loss_weight_charge_reg; sigdigits=4)
    weighted_charge_reg_train = epochs_mean_charge_reg_train .* plot_charge_reg_weight
    weighted_charge_reg_val   = epochs_mean_charge_reg_val   .* plot_charge_reg_weight
    plot_torsion_ks_weight = round(0.2 / loss_weight_torsion_ks; sigdigits=4)
    weighted_torsion_ks_train = epochs_mean_torsion_ks_train .* plot_torsion_ks_weight
    weighted_torsion_ks_val   = epochs_mean_torsion_ks_val   .* plot_torsion_ks_weight
    plot_enth_vap_weight = round(0.05 / loss_weight_enth_vap; sigdigits=4)
    weighted_enth_vap_train = epochs_mean_enth_vap_train .* plot_enth_vap_weight
    weighted_enth_vap_val   = epochs_mean_enth_vap_val   .* plot_enth_vap_weight
    plot_enth_mixing_weight = round(0.1 / loss_weight_enth_mixing; sigdigits=4)
    weighted_enth_mixing_train = epochs_mean_enth_mixing_train .* plot_enth_mixing_weight
    weighted_enth_mixing_val   = epochs_mean_enth_mixing_val   .* plot_enth_mixing_weight
    plot_J_coupling_weight = round(0.02 / loss_weight_J_coupling; sigdigits=4)
    weighted_J_coupling_train = epochs_mean_J_coupling_train .* plot_J_coupling_weight
    weighted_J_coupling_val   = epochs_mean_J_coupling_val   .* plot_J_coupling_weight
    plot_chem_shift_weight = round(0.25 / loss_weight_chem_shift; sigdigits=4)
    weighted_chem_shift_train = epochs_mean_chem_shift_train .* plot_chem_shift_weight
    weighted_chem_shift_val   = epochs_mean_chem_shift_val   .* plot_chem_shift_weight

    lines!(ax, epochs_mean_fs_intra_train,
           color=:blue, linestyle=:dash)
    lines!(ax, epochs_mean_fs_intra_val, label="Forces intra val",
           color=:blue)
    lines!(ax, epochs_mean_fs_intra_train_gems,
           color=:lightblue, linestyle=:dash)
    lines!(ax, epochs_mean_fs_intra_val_gems, label="Forces intra val GEMS",
           color=:lightblue)
    lines!(ax, weighted_fs_inter_train,
           color=:green, linestyle=:dash)
    lines!(ax, weighted_fs_inter_val, label="Forces inter val * $plot_fs_inter_weight",
           color=:green)
    lines!(ax, weighted_fs_inter_train_gems,
           color=:lightgreen, linestyle=:dash)
    lines!(ax, weighted_fs_inter_val_gems, label="Forces inter val GEMS * $plot_fs_inter_weight",
           color=:lightgreen)
    lines!(ax, weighted_pe_train,
           color=:red, linestyle=:dash)
    lines!(ax, weighted_pe_val, label="Energy val * $plot_pe_weight",
           color=:red)
    lines!(ax, weighted_charges_train,
           color=:orange, linestyle=:dash)
    lines!(ax, weighted_charges_val, label="Charge val * $plot_charge_weight",
           color=:orange)
    lines!(ax, weighted_enth_vap_train,
           color=:cyan, linestyle=:dash)
    lines!(ax, weighted_enth_vap_val, label="ΔHvap val * $plot_enth_vap_weight",
           color=:cyan)
    lines!(ax, weighted_enth_mixing_train,
           color=:magenta, linestyle=:dash)
    lines!(ax, weighted_enth_mixing_val, label="ΔHmix val * $plot_enth_mixing_weight",
           color=:magenta)
    lines!(ax, weighted_J_coupling_train,
           color=:brown, linestyle=:dash)
    lines!(ax, weighted_J_coupling_val, label="J-coupling val * $plot_J_coupling_weight",
           color=:brown)
    lines!(ax, weighted_chem_shift_train,
           color=:purple, linestyle=:dash)
    lines!(ax, weighted_chem_shift_val, label="Chemical shift val * $plot_chem_shift_weight",
           color=:purple)
    if loss_weight_charge_reg > 0
        lines!(ax, weighted_charge_reg_train,
            color=:grey, linestyle=:dash)
        lines!(ax, weighted_charge_reg_val, label="charge reg val * $plot_charge_reg_weight",
            color=:grey)
    end
    lines!(ax, weighted_torsion_ks_train,
           color=:pink, linestyle=:dash)
    lines!(ax, weighted_torsion_ks_val, label="Torsion ks val * $plot_torsion_ks_weight",
           color=:pink)
    #=lines!(ax, epochs_loss_regularisation, label="Regularisation",
           color=:pink)=#
    y_max = 1.2
    if training_sims_first_epoch > 0
        lines!(ax, [training_sims_first_epoch, training_sims_first_epoch], [0.0, y_max],
               color=:grey, label="Training simulations start")
    end
    xlims!(ax, low=0.0)
    ylims!(ax, low=0.0, high=y_max)
    f[1, 3] = Legend(f, ax; framevisible=false)

    coords = zeros(SVector{3, T}, 9906)
    sys, _, _, _, _ = mol_to_system(mol_id, mol_features_prot["protein_gb3_full"], coords,
                                       boundary_inf, models...)
    θs = collect(-π:(π/100):(π+0.001))
    ind_ϕ, ind_ψ = 223, 262
    proper_ind = findfirst(i -> isa(i, InteractionList4Atoms), sys.specific_inter_lists)
    if !isnothing(proper_ind) && use_proptor
        propers = sys.specific_inter_lists[proper_ind]
        @assert getindex.((propers.is, propers.js, propers.ks, propers.ls), ind_ϕ) == (82, 99, 100, 101)
        @assert getindex.((propers.is, propers.js, propers.ks, propers.ls), ind_ψ) == (99, 100, 101, 115)
        learned_pes_ϕ = torsion_pes(propers.inters[ind_ϕ], θs)
        learned_pes_ψ = torsion_pes(propers.inters[ind_ψ], θs)
        ff99SBildn_ϕ = PeriodicTorsion(
            periodicities=(3, 2),
            phases=(zero(T), zero(T)),
            ks=(T(1.75728), T(1.12968)),
        )
        ff99SBildn_ψ = PeriodicTorsion(
            periodicities=(3, 2, 1),
            phases=(T(π), T(π), T(π)),
            ks=(T(2.3012), T(6.61072), T(1.8828)),
        )
        ff99SBildn_pes_ϕ = torsion_pes(ff99SBildn_ϕ, θs)
        ff99SBildn_pes_ψ = torsion_pes(ff99SBildn_ψ, θs)

        ax1 = Axis(f[2, 1], xlabel="ϕ angle / radians", ylabel="Potential energy / kJ/mol")
        lines!(ax1, θs, learned_pes_ϕ, label="Learned potential")
        lines!(ax1, θs, ff99SBildn_pes_ϕ, label="ff99SBildn")
        ax2 = Axis(f[2, 2], xlabel="ψ angle / radians", ylabel="Potential energy / kJ/mol")
        lines!(ax2, θs, learned_pes_ψ, label="Learned potential")
        lines!(ax2, θs, ff99SBildn_pes_ψ, label="ff99SBildn")
        f[2, 3] = Legend(f, ax1; framevisible=false)
    end
    save(plot_fp, f)
end

function read_conf_data(mol_order, start_i, end_i)
    hdf5_list = [h5open(joinpath(data_dir, hdf5_file), "r") for hdf5_file in hdf5_files]
    map_res = map(start_i:end_i) do i
        mol_id, conf_i, conf_i_p1, repeat_i = mol_order[i]
        mol_hdf5_or_xyz, _, _ = extract_hdf5_or_xyz(mol_id, hdf5_list)
        coords = read_coordinates(mol_hdf5_or_xyz, conf_i)
        dft_fs, exceeds_max = read_dft_forces(mol_hdf5_or_xyz, conf_i)
        dft_pe = read_dft_pe(mol_hdf5_or_xyz, conf_i)
        dft_charges, has_charges = read_dft_charges(mol_hdf5_or_xyz, conf_i)
        pair_present = !iszero(conf_i_p1)
        if pair_present
            coords_p1 = read_coordinates(mol_hdf5_or_xyz, conf_i_p1)
            dft_fs_p1, exceeds_max_p1 = read_dft_forces(mol_hdf5_or_xyz, conf_i_p1)
            dft_pe_p1 = read_dft_pe(mol_hdf5_or_xyz, conf_i_p1)
            dft_charges_p1, has_charges_p1 = read_dft_charges(mol_hdf5_or_xyz, conf_i_p1)
        else
            coords_p1, dft_fs_p1, exceeds_max_p1, dft_pe_p1, dft_charges_p1, has_charges_p1 = coords,
                                dft_fs, exceeds_max, dft_pe, dft_charges, has_charges
        end
        exceeds_max_either = exceeds_max || exceeds_max_p1
        return coords, dft_fs, dft_pe, dft_charges, has_charges, coords_p1, dft_fs_p1,
                dft_pe_p1, dft_charges_p1, has_charges_p1, exceeds_max_either, pair_present
    end
    close.(hdf5_list)
    return map_res
end

function calc_mean_U_gas(mol_id, training_sim_dir, temp, models...)
    frame_is = ignore_derivatives() do
        shuffle(cond_sim_frames)[1:enth_vap_gas_n_samples]
    end
    pe_sum = zero(T)
    for frame_i in frame_is
        coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
        _, pe, _, _, _, _ = mol_to_preds(
                        mol_id, mol_features_cond[mol_id], coords, boundary, models...)
        pe_sum += pe
    end
    return pe_sum / enth_vap_gas_n_samples
end

function read_traj_chem_shifts(training_sim_dir)
    d = Dict{String, Vector{Vector{T}}}()
    for protein in keys(protein_data)
        fp = joinpath(training_sim_dir, "protein", "$(protein)_chem_shifts.txt")
        chem_shifts = [parse.(T, split(l)) for l in readlines(fp)]
        @assert length(chem_shifts) == maximum(prot_sim_frames)
        @assert all(isequal(protein_data[protein].n_res * 4 - 3), map(length, chem_shifts))
        d[protein] = chem_shifts
    end
    return d
end

function read_traj_pes(training_sim_dir)
    d = Dict{String, Vector{T}}()
    for protein in keys(protein_data)
        fp = joinpath(training_sim_dir, "protein", "$protein.log")
        pes = [parse(T, split(l, ",")[2]) for l in readlines(fp)[2:end]]
        @assert length(pes) == maximum(prot_sim_frames)
        d[protein] = pes
    end
    return d
end

function train_epoch!(models, optims, epoch_n,
                      epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
                      epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
                      epochs_mean_pe_train, epochs_mean_pe_val,
                      epochs_mean_charges_train, epochs_mean_charges_val,
                      epochs_mean_charge_reg_train, epochs_mean_charge_reg_val,
                      epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
                      epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
                      epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems,
                      epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
                      epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
                      epochs_mean_J_coupling_train, epochs_mean_J_coupling_val,
                      epochs_mean_chem_shift_train, epochs_mean_chem_shift_val,
                      epochs_loss_regularisation)
    time_start = now()
    # Energy pairs changed every epoch
    train_order = shuffle(read_conformations(molecules_train))
    val_order = read_conformations(molecules_val)
    n_conf_pairs_train, n_conf_pairs_val = length(train_order), length(val_order)
    n_batches_train = cld(n_conf_pairs_train, n_minibatch)
    n_batches_val   = cld(n_conf_pairs_val  , n_minibatch)
    train_order_gems = shuffle(repeat(molecules_train_gems, subset_n_repeats["GEMS crambin"]))
    val_order_gems   = shuffle(repeat(molecules_val_gems  , subset_n_repeats["GEMS crambin"]))
    train_order_cond, val_order_cond = shuffle(molecules_train_cond), shuffle(molecules_val_cond)
    time_wait_sims, time_spice, time_gems, time_cond, time_protein = zero(Float64), zero(Float64), zero(Float64), zero(Float64), zero(Float64)

    if !iszero(training_sims_first_epoch) && epoch_n >= (training_sims_first_epoch - 1)
        submit_dir = joinpath(out_dir, "training_sims", "epoch_$(epoch_n-1)")
        log_fp = joinpath(out_dir, "training_sims", "epoch_$(epoch_n-1).out")
        ff_xml_fp = joinpath(out_dir, "ff_xml", "epoch_$(epoch_n-1).xml")
        features_to_ff_xml(ff_xml_fp, mol_ids_xml, models...)
        submit_training_sims(submit_dir, ff_xml_fp, log_fp, epoch_n - 1)
    end
    if !iszero(training_sims_first_epoch) && epoch_n >= training_sims_first_epoch
        if epoch_n >= (training_sims_first_epoch + 1)
            rm(joinpath(out_dir, "training_sims", "epoch_$(epoch_n-3)");
               recursive=true, force=true)
        end
        training_sim_dir = joinpath(out_dir, "training_sims", "epoch_$(epoch_n-2)")
        training_sims_complete = false
        time_group = time()
        while !training_sims_complete
            if isfile(joinpath(training_sim_dir, "done.txt"))
                training_sims_complete = true
            else
                sleep(10)
            end
        end
        time_wait_sims += time() - time_group
        simulation_str = "used simulations from end of epoch $(epoch_n-2)"
    elseif use_ref_simulations
        training_sim_dir = ref_traj_dir
        simulation_str = "used simulations from GAFF/Amber14/TIP3P"
    else
        training_sim_dir = ""
        simulation_str = "did not use simulations"
    end
    if training_sim_dir != ""
        traj_chem_shifts = read_traj_chem_shifts(training_sim_dir)
        traj_pes = read_traj_pes(training_sim_dir)
    end

    loss_sum_fs_intra_train, loss_sum_fs_intra_val = zero(T), zero(T)
    loss_sum_fs_inter_train, loss_sum_fs_inter_val = zero(T), zero(T)
    loss_sum_pe_train, loss_sum_pe_val = zero(T), zero(T)
    loss_sum_charges_train, loss_sum_charges_val = zero(T), zero(T)
    loss_sum_charge_reg_train, loss_sum_charge_reg_val = zero(T), zero(T)
    loss_sum_torsion_ks_train, loss_sum_torsion_ks_val = zero(T), zero(T)
    loss_sum_fs_intra_train_gems, loss_sum_fs_intra_val_gems = zero(T), zero(T)
    loss_sum_fs_inter_train_gems, loss_sum_fs_inter_val_gems = zero(T), zero(T)
    loss_sum_enth_vap_train, loss_sum_enth_vap_val = zero(T), zero(T)
    loss_sum_enth_mixing_train, loss_sum_enth_mixing_val = zero(T), zero(T)
    loss_sum_J_coupling_train, loss_sum_J_coupling_val = zero(T), zero(T)
    loss_sum_chem_shift_train, loss_sum_chem_shift_val = zero(T), zero(T)
    count_confs_train, count_confs_val = 0, 0
    count_confs_inter_train, count_confs_inter_val = 0, 0
    count_confs_pe_train, count_confs_pe_val = 0, 0
    count_confs_charges_train, count_confs_charges_val = 0, 0
    count_confs_torsion_ks_train, count_confs_torsion_ks_val = 0, 0
    count_confs_train_gems, count_confs_val_gems = 0, 0
    count_confs_inter_train_gems, count_confs_inter_val_gems = 0, 0
    count_confs_enth_vap_train, count_confs_enth_vap_val = 0, 0
    count_confs_enth_mixing_train, count_confs_enth_mixing_val = 0, 0
    count_confs_J_coupling_train, count_confs_J_coupling_val = 0, 0
    loss_jc, loss_cs = zero(T), zero(T)
    grads_jc = convert(Vector{Any}, fill(nothing, length(models)))
    grads_cs = convert(Vector{Any}, fill(nothing, length(models)))

    n_chunks = Threads.nthreads()
    if !isnothing(out_dir)
        for store_id in ("val-val", "ΔHvap", "ΔHmix")
            rm(joinpath(out_dir, "store_$store_id.txt"); force=true)
        end
    end

    Flux.trainmode!(models)
    for batch_i in 1:n_batches_train
        start_i = (batch_i - 1) * n_minibatch + 1
        end_i = min(start_i + n_minibatch - 1, n_conf_pairs_train)
        loss_fs_intra_chunks      = [T[] for _ in 1:n_chunks]
        loss_fs_inter_chunks      = [T[] for _ in 1:n_chunks]
        loss_pe_chunks            = [T[] for _ in 1:n_chunks]
        loss_charges_chunks       = [T[] for _ in 1:n_chunks]
        loss_charge_reg_chunks    = [T[] for _ in 1:n_chunks]
        loss_torsion_ks_chunks    = [T[] for _ in 1:n_chunks]
        loss_fs_intra_chunks_gems = [T[] for _ in 1:n_chunks]
        loss_fs_inter_chunks_gems = [T[] for _ in 1:n_chunks]
        loss_enth_vap_chunks      = [T[] for _ in 1:n_chunks]
        loss_enth_mixing_chunks   = [T[] for _ in 1:n_chunks]
        loss_J_coupling_chunks    = [T[] for _ in 1:n_chunks]
        loss_chem_shift_chunks    = [T[] for _ in 1:n_chunks]
        grads_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
        print_chunks = fill("", n_chunks)
        conf_data = read_conf_data(train_order, start_i, end_i)

        time_group = time()
        @timeit to "SPICE" Threads.@threads for chunk_id in 1:n_chunks
            for i in (start_i - 1 + chunk_id):n_chunks:end_i
                mol_id, conf_i, conf_i_p1, repeat_i = train_order[i]
                coords, dft_fs, dft_pe, dft_charges, has_charges, coords_p1, dft_fs_p1, dft_pe_p1,
                        dft_charges_p1, has_charges_p1, exceeds_max_either,
                        pair_present = conf_data[i - start_i + 1]
                if verbose
                    # This only prints for the first of each pair and does not print energy
                    print_chunks[chunk_id] *= "$mol_id conf $conf_i training - "
                end
                if exceeds_max_either
                    if verbose
                        print_chunks[chunk_id] *= "max force exceeded\n"
                    end
                    continue
                end

                grads = Zygote.gradient(models...) do models...
                    loss_fs_intra_grad_sum, loss_fs_inter_grad_sum = zero(T), zero(T)
                    loss_pe_sum, loss_charges_sum, loss_charge_reg_sum, loss_torsion_ks_sum, loss_regularisation_sum = zero(T), zero(T), zero(T), zero(T), zero(T)

                    fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                    mol_id, mol_features[mol_id], coords, boundary_inf, models...)
                    fs_intra, fs_inter = split_forces(fs, coords, molecule_inds, elements)
                    dft_fs_intra, dft_fs_inter = split_forces(dft_fs, coords, molecule_inds, elements)
                    if mse_loss
                        loss_fs_intra_grad = force_loss_mse(fs_intra, dft_fs_intra)
                        loss_fs_inter_grad = force_loss_mse(fs_inter, dft_fs_inter) * loss_weight_force_inter
                        loss_fs_intra_print = force_loss(fs_intra, dft_fs_intra)
                        loss_fs_inter_print = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                    else
                        loss_fs_intra_grad = force_loss(fs_intra, dft_fs_intra)
                        loss_fs_inter_grad = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                        loss_fs_intra_print, loss_fs_inter_print = loss_fs_intra_grad, loss_fs_inter_grad
                    end
                    loss_charges = (has_charges ? charge_loss(charges, dft_charges) : zero(T))
                    loss_charge_reg = charge_regularisation(charges)
                    loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                    loss_regularisation = param_regularisation((models...,))
                    if verbose
                        ignore_derivatives() do
                            print_chunks[chunk_id] *= "loss forces intra $loss_fs_intra_print forces inter $loss_fs_inter_print charge $loss_charges charge reg $loss_charge_reg torsion ks $loss_torsion_ks regularisation $loss_regularisation\n"
                        end
                    end
                    if isnan(loss_fs_intra_grad) || isnan(loss_fs_intra_print) ||
                                        isnan(loss_fs_inter_grad) || isnan(loss_fs_inter_print) ||
                                        isnan(loss_charges) || isnan(loss_charge_reg) ||
                                        isnan(loss_torsion_ks) || isnan(loss_regularisation)
                        return zero(T)
                    else
                        ignore_derivatives() do
                            push!(loss_fs_intra_chunks[chunk_id], loss_fs_intra_print)
                            push!(loss_fs_inter_chunks[chunk_id], loss_fs_inter_print)
                            push!(loss_charges_chunks[chunk_id], loss_charges)
                            push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                            push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                        end
                        loss_fs_intra_grad_sum += loss_fs_intra_grad
                        loss_fs_inter_grad_sum += loss_fs_inter_grad
                        loss_charges_sum += loss_charges
                        loss_charge_reg_sum += loss_charge_reg
                        loss_torsion_ks_sum += loss_torsion_ks
                        loss_regularisation_sum += loss_regularisation
                    end

                    if pair_present
                        fs, pe_p1, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                    mol_id, mol_features[mol_id], coords_p1, boundary_inf, models...)
                        fs_intra, fs_inter = split_forces(fs, coords_p1, molecule_inds, elements)
                        dft_fs_intra, dft_fs_inter = split_forces(dft_fs_p1, coords_p1, molecule_inds, elements)
                        if mse_loss
                            loss_fs_intra_grad = force_loss_mse(fs_intra, dft_fs_intra)
                            loss_fs_inter_grad = force_loss_mse(fs_inter, dft_fs_inter) * loss_weight_force_inter
                            loss_fs_intra_print = force_loss(fs_intra, dft_fs_intra)
                            loss_fs_inter_print = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                        else
                            loss_fs_intra_grad = force_loss(fs_intra, dft_fs_intra)
                            loss_fs_inter_grad = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                            loss_fs_intra_print, loss_fs_inter_print = loss_fs_intra_grad, loss_fs_inter_grad
                        end
                        loss_charges = (has_charges_p1 ? charge_loss(charges, dft_charges_p1) : zero(T))
                        loss_charge_reg = charge_regularisation(charges)
                        loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                        loss_regularisation = param_regularisation((models...,))
                        if isnan(loss_fs_intra_grad) || isnan(loss_fs_intra_print) ||
                                            isnan(loss_fs_inter_grad) || isnan(loss_fs_inter_print)
                                            isnan(loss_charges) || isnan(loss_charge_reg) ||
                                            isnan(loss_torsion_ks) || isnan(loss_regularisation)
                            return zero(T)
                        else
                            if epoch_n >= loss_energy_first_epoch
                                pe_diff = pe_p1 - pe
                                dft_pe_diff = dft_pe_p1 - dft_pe
                                if mse_loss
                                    loss_pe_unbound = pe_loss_mse(pe_diff, dft_pe_diff)
                                else
                                    loss_pe_unbound = pe_loss(pe_diff, dft_pe_diff)
                                end
                                if loss_pe_unbound < loss_energy_max
                                    loss_pe = loss_pe_unbound
                                else
                                    loss_pe = zero(T)
                                end
                            else
                                loss_pe = zero(T)
                            end
                            if isnan(loss_pe)
                                return zero(T)
                            end

                            ignore_derivatives() do
                                push!(loss_fs_intra_chunks[chunk_id], loss_fs_intra_print)
                                push!(loss_fs_inter_chunks[chunk_id], loss_fs_inter_print)
                                push!(loss_pe_chunks[chunk_id], loss_pe)
                                push!(loss_charges_chunks[chunk_id], loss_charges)
                                push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                                push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                            end
                            loss_fs_intra_grad_sum += loss_fs_intra_grad
                            loss_fs_inter_grad_sum += loss_fs_inter_grad
                            loss_pe_sum += loss_pe
                            loss_charges_sum += loss_charges
                            loss_charge_reg_sum += loss_charge_reg
                            loss_torsion_ks_sum += loss_torsion_ks
                            loss_regularisation_sum += loss_regularisation
                        end
                    end
                    return loss_fs_intra_grad_sum * train_on_fs_intra +
                           loss_fs_inter_grad_sum * train_on_fs_inter +
                           loss_pe_sum * train_on_pe +
                           loss_charges_sum * train_on_charges +
                           loss_charge_reg_sum * train_on_charge_reg +
                           loss_torsion_ks_sum +
                           loss_regularisation_sum
                end

                if check_no_nans(grads)
                    grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads)
                end
            end
        end
        verbose && foreach(report, print_chunks)
        print_chunks = fill("", n_chunks)
        time_spice += time() - time_group

        time_group = time()
        gems_inds = collect(batch_i:n_batches_train:length(train_order_gems))
        @timeit to "GEMS" Threads.@threads for chunk_id in 1:n_chunks
            for gems_inds_i in chunk_id:n_chunks:length(gems_inds)
                gems_i = gems_inds[gems_inds_i]
                mol_id = train_order_gems[gems_i]
                conf_i = parse(Int, split(mol_id, "_")[2])
                coords, dft_fs = read_coordinates_gems(conf_i), read_dft_forces_gems(conf_i)
                if verbose
                    print_chunks[chunk_id] *= "$mol_id training - "
                end

                grads = Zygote.gradient(models...) do models...
                    fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                    mol_id, mol_features_gems[mol_id], coords, boundary_inf, models...)
                    fs_intra, fs_inter = split_forces(fs, coords, molecule_inds, elements)
                    dft_fs_intra, dft_fs_inter = split_forces(dft_fs, coords, molecule_inds, elements)
                    if mse_loss
                        loss_fs_intra_grad = force_loss_mse(fs_intra, dft_fs_intra)
                        loss_fs_inter_grad = force_loss_mse(fs_inter, dft_fs_inter) * loss_weight_force_inter
                        loss_fs_intra_print = force_loss(fs_intra, dft_fs_intra)
                        loss_fs_inter_print = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                    else
                        loss_fs_intra_grad = force_loss(fs_intra, dft_fs_intra)
                        loss_fs_inter_grad = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                        loss_fs_intra_print, loss_fs_inter_print = loss_fs_intra_grad, loss_fs_inter_grad
                    end
                    loss_charge_reg = charge_regularisation(charges)
                    loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                    loss_regularisation = param_regularisation((models...,))
                    if verbose
                        ignore_derivatives() do
                            print_chunks[chunk_id] *= "loss forces intra $loss_fs_intra_print forces inter $loss_fs_inter_print\n"
                        end
                    end
                    if isnan(loss_fs_intra_grad) || isnan(loss_fs_intra_print) ||
                                        isnan(loss_fs_inter_grad) || isnan(loss_fs_inter_print) ||
                                        isnan(loss_charge_reg) || isnan(loss_torsion_ks) ||
                                        isnan(loss_regularisation)
                        return zero(T)
                    else
                        ignore_derivatives() do
                            push!(loss_fs_intra_chunks_gems[chunk_id], loss_fs_intra_print)
                            push!(loss_fs_inter_chunks_gems[chunk_id], loss_fs_inter_print)
                            push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                            push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                        end
                    end
                    return loss_fs_intra_grad * train_on_fs_intra +
                           loss_fs_inter_grad * train_on_fs_inter +
                           loss_charge_reg * train_on_charge_reg +
                           loss_torsion_ks +
                           loss_regularisation
                end

                if check_no_nans(grads)
                    grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads)
                end
            end
        end
        verbose && foreach(report, print_chunks)
        print_chunks = fill("", n_chunks)
        time_gems += time() - time_group

        time_group = time()
        if training_sim_dir != "" && (loss_weight_enth_vap > zero(T) || loss_weight_enth_mixing > zero(T))
            cond_inds = collect(batch_i:n_batches_train:length(train_order_cond))
            @timeit to "Condensed" #=Threads.@threads=# for chunk_id in 1:n_chunks
                for cond_inds_i in chunk_id:n_chunks:length(cond_inds)
                    cond_i = cond_inds[cond_inds_i]
                    mol_id, temp, frame_i, repeat_i = train_order_cond[cond_i]
                    if startswith(mol_id, "vapourisation_")
                        train_on_weight = train_on_enth_vap
                        label = "ΔHvap"
                    else
                        train_on_weight = train_on_enth_mixing
                        label = "ΔHmix"
                    end
                    if verbose
                        print_chunks[chunk_id] *= "$mol_id training - "
                    end

                    grads = Zygote.gradient(models...) do models...
                        if startswith(mol_id, "vapourisation_")
                            coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
                            _, pe, charges, torsion_ks_size, _, molecule_inds = mol_to_preds(
                                    mol_id, mol_features_cond[mol_id], coords, boundary, models...)
                            mol_id_gas = ignore_derivatives() do
                                replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                            end
                            mean_U_gas = calc_mean_U_gas(mol_id_gas, training_sim_dir, temp, models...)
                            loss_cond = enth_vap_loss(pe, mean_U_gas, temp, frame_i, repeat_i,
                                                      maximum(molecule_inds), mol_id)
                        else
                            _, _, smiles_1, smiles_2 = split_grad_safe(mol_id, "_")
                            mol_id_1 = "mixing_single_$smiles_1"
                            mol_id_2 = "mixing_single_$smiles_2"
                            coords_1, boundary_1 = read_sim_data(mol_id_1, training_sim_dir, frame_i)
                            _, pe_1, _, _, _, molecule_inds_1 = mol_to_preds(
                                mol_id_1, mol_features_cond[mol_id_1], coords_1, boundary_1, models...)
                            coords_2, boundary_2 = read_sim_data(mol_id_2, training_sim_dir, frame_i)
                            _, pe_2, _, _, _, molecule_inds_2 = mol_to_preds(
                                mol_id_2, mol_features_cond[mol_id_2], coords_2, boundary_2, models...)
                            coords_com, boundary_com = read_sim_data(mol_id, training_sim_dir, frame_i)
                            _, pe_com, charges, torsion_ks_size, _, molecule_inds_com = mol_to_preds(
                                mol_id, mol_features_cond[mol_id], coords_com, boundary_com, models...)
                            loss_cond = enth_mixing_loss(pe_com, pe_1, pe_2, boundary_com,
                                            boundary_1, boundary_2, maximum(molecule_inds_com),
                                            maximum(molecule_inds_1), maximum(molecule_inds_2),
                                            mol_id, frame_i, repeat_i)
                        end
                        loss_charge_reg = charge_regularisation(charges)
                        loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                        loss_regularisation = param_regularisation((models...,))
                        if verbose
                            ignore_derivatives() do
                                print_chunks[chunk_id] *= "loss $label $loss_cond\n"
                            end
                        end
                        if isnan(loss_cond) || isnan(loss_charge_reg) ||
                                        isnan(loss_torsion_ks) || isnan(loss_regularisation)
                            return zero(T)
                        else
                            ignore_derivatives() do
                                if startswith(mol_id, "vapourisation_")
                                    push!(loss_enth_vap_chunks[chunk_id], loss_cond)
                                else
                                    push!(loss_enth_mixing_chunks[chunk_id], loss_cond)
                                end
                                push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                                push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                            end
                        end
                        return loss_cond * train_on_weight +
                               loss_charge_reg * train_on_charge_reg +
                               loss_torsion_ks +
                               loss_regularisation
                    end

                    if check_no_nans(grads)
                        grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads)
                    end
                end
            end
            verbose && foreach(report, print_chunks)
        end
        time_cond += time() - time_group

        time_group = time()
        if training_sim_dir != "" &&
                    (loss_weight_J_coupling > zero(T) || loss_weight_chem_shift > zero(T))
            # Threading used in function
            @timeit to "Protein" for protein in keys(protein_data)
                if (batch_i - 1) % prot_reweighting_n_batches == 0
                    if iszero(prot_reweighting_n_samples)
                        frames_sample = frames_prot_train
                    else
                        frames_sample = shuffle(frames_prot_train)[1:prot_reweighting_n_samples]
                    end
                    loss_jc, loss_cs, grads_jc, grads_cs = protein_loss_grads("protein_$protein",
                                training_sim_dir, traj_pes, traj_chem_shifts,
                                frames_sample, true, models...)
                end
                if verbose
                    report("loss J-Coupling $loss_jc chemical shift $loss_cs\n")
                end
                if !isnan(loss_jc) && !isnan(loss_cs)
                    chunk_id = 1
                    push!(loss_J_coupling_chunks[chunk_id], loss_jc)
                    push!(loss_chem_shift_chunks[chunk_id], loss_cs)
                    if check_no_nans(grads_jc) && train_on_J_coupling == one(T)
                        grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads_jc)
                    end
                    if check_no_nans(grads_cs) && train_on_chem_shifts == one(T)
                        grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads_cs)
                    end
                end
            end
        end
        time_protein += time() - time_group

        @timeit to "Gradients" begin
            grads_minibatch = convert(Vector{Any}, fill(nothing, length(models)))
            for chunk_id in 1:n_chunks
                grads_minibatch = accum_grads.(grads_minibatch, grads_chunks[chunk_id])
            end
            grad_vals, restructure = Flux.destructure(grads_minibatch)
            grads_clamped = restructure(clamp.(grad_vals, -grad_clamp_val, grad_clamp_val))
            for mi in eachindex(models)
                Flux.update!(optims[mi], models[mi], grads_clamped[mi])
            end
        end

        @timeit to "Logging" begin
            losses_fs_intra      = vcat(loss_fs_intra_chunks...)
            losses_fs_inter      = vcat(loss_fs_inter_chunks...)
            losses_pe            = vcat(loss_pe_chunks...)
            losses_charges       = vcat(loss_charges_chunks...)
            losses_charge_reg    = vcat(loss_charge_reg_chunks...)
            losses_torsion_ks    = vcat(loss_torsion_ks_chunks...)
            losses_fs_intra_gems = vcat(loss_fs_intra_chunks_gems...)
            losses_fs_inter_gems = vcat(loss_fs_inter_chunks_gems...)
            losses_enth_vap      = vcat(loss_enth_vap_chunks...)
            losses_enth_mixing   = vcat(loss_enth_mixing_chunks...)
            losses_J_coupling    = vcat(loss_J_coupling_chunks...)
            losses_chem_shift    = vcat(loss_chem_shift_chunks...)
            if length(losses_fs_intra) > 0
                loss_sum_fs_intra_train      += sum(losses_fs_intra)
                loss_sum_fs_inter_train      += sum(losses_fs_inter)
                loss_sum_pe_train            += sum(losses_pe)
                loss_sum_charges_train       += sum(losses_charges)
                loss_sum_charge_reg_train    += sum(losses_charge_reg)
                loss_sum_torsion_ks_train    += sum(losses_torsion_ks)
                loss_sum_fs_intra_train_gems += sum(losses_fs_intra_gems)
                loss_sum_fs_inter_train_gems += sum(losses_fs_inter_gems)
                loss_sum_enth_vap_train      += sum(losses_enth_vap)
                loss_sum_enth_mixing_train   += sum(losses_enth_mixing)
                loss_sum_J_coupling_train    += sum(losses_J_coupling)
                loss_sum_chem_shift_train    += sum(losses_chem_shift)

                count_confs_train             += length(losses_fs_intra)
                count_confs_inter_train       += count(!iszero, losses_fs_inter)
                count_confs_pe_train          += length(losses_pe)
                count_confs_charges_train     += count(!iszero, losses_charges)
                count_confs_torsion_ks_train  += length(losses_torsion_ks)
                count_confs_train_gems        += length(losses_fs_intra_gems)
                count_confs_inter_train_gems  += count(!iszero, losses_fs_inter_gems)
                count_confs_enth_vap_train    += length(losses_enth_vap)
                count_confs_enth_mixing_train += length(losses_enth_mixing)
                count_confs_J_coupling_train  += length(losses_J_coupling)
            end
        end
    end

    Flux.testmode!(models)
    for batch_i in 1:n_batches_val
        start_i = (batch_i - 1) * n_minibatch + 1
        end_i = min(start_i + n_minibatch - 1, n_conf_pairs_val)
        loss_fs_intra_chunks      = [T[] for _ in 1:n_chunks]
        loss_fs_inter_chunks      = [T[] for _ in 1:n_chunks]
        loss_pe_chunks            = [T[] for _ in 1:n_chunks]
        loss_charges_chunks       = [T[] for _ in 1:n_chunks]
        loss_charge_reg_chunks    = [T[] for _ in 1:n_chunks]
        loss_torsion_ks_chunks    = [T[] for _ in 1:n_chunks]
        loss_fs_intra_chunks_gems = [T[] for _ in 1:n_chunks]
        loss_fs_inter_chunks_gems = [T[] for _ in 1:n_chunks]
        loss_enth_vap_chunks      = [T[] for _ in 1:n_chunks]
        loss_enth_mixing_chunks   = [T[] for _ in 1:n_chunks]
        loss_J_coupling_chunks    = [T[] for _ in 1:n_chunks]
        loss_chem_shift_chunks    = [T[] for _ in 1:n_chunks]
        print_chunks = fill("", n_chunks)
        conf_data = read_conf_data(val_order, start_i, end_i)

        Threads.@threads for chunk_id in 1:n_chunks
            for i in (start_i - 1 + chunk_id):n_chunks:end_i
                mol_id, conf_i, conf_i_p1, repeat_i = val_order[i]
                coords, dft_fs, dft_pe, dft_charges, has_charges, coords_p1, dft_fs_p1, dft_pe_p1,
                        dft_charges_p1, has_charges_p1, exceeds_max_either,
                        pair_present = conf_data[i - start_i + 1]
                if verbose
                    print_chunks[chunk_id] *= "$mol_id conf $conf_i validation - "
                end
                if exceeds_max_either
                    if verbose
                        print_chunks[chunk_id] *= "max force exceeded\n"
                    end
                    continue
                end

                fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                mol_id, mol_features[mol_id], coords, boundary_inf, models...)
                fs_intra, fs_inter = split_forces(fs, coords, molecule_inds, elements)
                dft_fs_intra, dft_fs_inter = split_forces(dft_fs, coords, molecule_inds, elements)
                loss_fs_intra = force_loss(fs_intra, dft_fs_intra)
                loss_fs_inter = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                loss_charges = (has_charges ? charge_loss(charges, dft_charges) : zero(T))
                loss_charge_reg = charge_regularisation(charges)
                loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                if verbose
                    print_chunks[chunk_id] *= "loss forces intra $loss_fs_intra forces inter $loss_fs_inter charge $loss_charges charge reg $loss_charge_reg torsion ks $loss_torsion_ks\n"
                end
                if !(isnan(loss_fs_intra) || isnan(loss_fs_inter) || isnan(loss_charges) ||
                                                isnan(loss_charge_reg) || isnan(loss_torsion_ks))
                    push!(loss_fs_intra_chunks[chunk_id], loss_fs_intra)
                    push!(loss_fs_inter_chunks[chunk_id], loss_fs_inter)
                    push!(loss_charges_chunks[chunk_id], loss_charges)
                    push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                    push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                end

                if pair_present
                    fs, pe_p1, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                mol_id, mol_features[mol_id], coords_p1, boundary_inf, models...)
                    fs_intra, fs_inter = split_forces(fs, coords_p1, molecule_inds, elements)
                    dft_fs_intra, dft_fs_inter = split_forces(dft_fs_p1, coords_p1, molecule_inds, elements)
                    loss_fs_intra = force_loss(fs_intra, dft_fs_intra)
                    loss_fs_inter = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                    loss_charges = (has_charges_p1 ? charge_loss(charges, dft_charges_p1) : zero(T))
                    loss_charge_reg = charge_regularisation(charges)
                    loss_torsion_ks = torsion_ks_loss(torsion_ks_size)

                    if !(isnan(loss_fs_intra) || isnan(loss_fs_inter) || isnan(loss_charges) ||
                                                isnan(loss_charge_reg) || isnan(loss_torsion_ks))
                        # Record the energy loss even if it is not used for training this epoch
                        pe_diff = pe_p1 - pe
                        dft_pe_diff = dft_pe_p1 - dft_pe
                        loss_pe_unbound = pe_loss(pe_diff, dft_pe_diff)
                        if loss_pe_unbound < loss_energy_max
                            loss_pe = loss_pe_unbound
                        else
                            loss_pe = zero(T)
                        end
                        if !isnan(loss_pe)
                            push!(loss_fs_intra_chunks[chunk_id], loss_fs_intra)
                            push!(loss_fs_inter_chunks[chunk_id], loss_fs_inter)
                            push!(loss_pe_chunks[chunk_id], loss_pe)
                            push!(loss_charges_chunks[chunk_id], loss_charges)
                            push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                            push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                        end
                    end
                end
            end
        end
        verbose && foreach(report, print_chunks)
        print_chunks = fill("", n_chunks)

        gems_inds = collect(batch_i:n_batches_val:length(val_order_gems))
        Threads.@threads for chunk_id in 1:n_chunks
            for gems_inds_i in chunk_id:n_chunks:length(gems_inds)
                gems_i = gems_inds[gems_inds_i]
                mol_id = val_order_gems[gems_i]
                conf_i = parse(Int, split(mol_id, "_")[2])
                coords, dft_fs = read_coordinates_gems(conf_i), read_dft_forces_gems(conf_i)
                if verbose
                    print_chunks[chunk_id] *= "$mol_id validation - "
                end

                fs, pe, charges, torsion_ks_size, elements, molecule_inds = mol_to_preds(
                                mol_id, mol_features_gems[mol_id], coords, boundary_inf, models...)
                fs_intra, fs_inter = split_forces(fs, coords, molecule_inds, elements)
                dft_fs_intra, dft_fs_inter = split_forces(dft_fs, coords, molecule_inds, elements)
                if mse_loss
                    loss_fs_intra_grad = force_loss_mse(fs_intra, dft_fs_intra)
                    loss_fs_inter_grad = force_loss_mse(fs_inter, dft_fs_inter) * loss_weight_force_inter
                    loss_fs_intra_print = force_loss(fs_intra, dft_fs_intra)
                    loss_fs_inter_print = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                else
                    loss_fs_intra_grad = force_loss(fs_intra, dft_fs_intra)
                    loss_fs_inter_grad = force_loss(fs_inter, dft_fs_inter) * loss_weight_force_inter
                    loss_fs_intra_print, loss_fs_inter_print = loss_fs_intra_grad, loss_fs_inter_grad
                end
                loss_charge_reg = charge_regularisation(charges)
                loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                if verbose
                    print_chunks[chunk_id] *= "loss forces intra $loss_fs_intra_print forces inter $loss_fs_inter_print\n"
                end
                if !(isnan(loss_fs_intra_grad) || isnan(loss_fs_intra_print) ||
                                    isnan(loss_fs_inter_grad) || isnan(loss_fs_inter_print) ||
                                    isnan(loss_charge_reg) || isnan(loss_torsion_ks))
                    push!(loss_fs_intra_chunks_gems[chunk_id], loss_fs_intra_print)
                    push!(loss_fs_inter_chunks_gems[chunk_id], loss_fs_inter_print)
                    push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                    push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                end
            end
        end
        verbose && foreach(report, print_chunks)
        print_chunks = fill("", n_chunks)

        if training_sim_dir != "" && (loss_weight_enth_vap > zero(T) || loss_weight_enth_mixing > zero(T))
            cond_inds = collect(batch_i:n_batches_val:length(val_order_cond))
            #=Threads.@threads=# for chunk_id in 1:n_chunks
                for cond_inds_i in chunk_id:n_chunks:length(cond_inds)
                    cond_i = cond_inds[cond_inds_i]
                    mol_id, temp, frame_i, repeat_i = val_order_cond[cond_i]
                    label = (startswith(mol_id, "vapourisation_") ? "ΔHvap" : "ΔHmix")
                    if verbose
                        print_chunks[chunk_id] *= "$mol_id validation - "
                    end

                    if startswith(mol_id, "vapourisation_")
                        coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
                        _, pe, charges, torsion_ks_size, _, molecule_inds = mol_to_preds(
                                        mol_id, mol_features_cond[mol_id], coords, boundary, models...)
                        mol_id_gas = replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                        mean_U_gas = calc_mean_U_gas(mol_id_gas, training_sim_dir, temp, models...)
                        loss_cond = enth_vap_loss(pe, mean_U_gas, temp, frame_i, repeat_i,
                                                  maximum(molecule_inds), mol_id)
                    else
                        _, _, smiles_1, smiles_2 = split(mol_id, "_")
                        mol_id_1 = "mixing_single_$smiles_1"
                        mol_id_2 = "mixing_single_$smiles_2"
                        coords_1, boundary_1 = read_sim_data(mol_id_1, training_sim_dir, frame_i)
                        _, pe_1, _, _, _, molecule_inds_1 = mol_to_preds(
                            mol_id_1, mol_features_cond[mol_id_1], coords_1, boundary_1, models...)
                        coords_2, boundary_2 = read_sim_data(mol_id_2, training_sim_dir, frame_i)
                        _, pe_2, _, _, _, molecule_inds_2 = mol_to_preds(
                            mol_id_2, mol_features_cond[mol_id_2], coords_2, boundary_2, models...)
                        coords_com, boundary_com = read_sim_data(mol_id, training_sim_dir, frame_i)
                        _, pe_com, charges, torsion_ks_size, _, molecule_inds_com = mol_to_preds(
                            mol_id, mol_features_cond[mol_id], coords_com, boundary_com, models...)
                        loss_cond = enth_mixing_loss(pe_com, pe_1, pe_2, boundary_com,
                                        boundary_1, boundary_2, maximum(molecule_inds_com),
                                        maximum(molecule_inds_1), maximum(molecule_inds_2),
                                        mol_id, frame_i, repeat_i)
                    end
                    loss_charge_reg = charge_regularisation(charges)
                    loss_torsion_ks = torsion_ks_loss(torsion_ks_size)
                    if verbose
                        print_chunks[chunk_id] *= "loss $label $loss_cond\n"
                    end
                    if !(isnan(loss_cond) || isnan(loss_charge_reg) || isnan(loss_torsion_ks))
                        if startswith(mol_id, "vapourisation_")
                            push!(loss_enth_vap_chunks[chunk_id], loss_cond)
                        else
                            push!(loss_enth_mixing_chunks[chunk_id], loss_cond)
                        end
                        push!(loss_charge_reg_chunks[chunk_id], loss_charge_reg)
                        push!(loss_torsion_ks_chunks[chunk_id], loss_torsion_ks)
                    end
                end
            end
            verbose && foreach(report, print_chunks)
        end

        if training_sim_dir != "" &&
                    (loss_weight_J_coupling > zero(T) || loss_weight_chem_shift > zero(T))
            for protein in keys(protein_data)
                # Since the same frames are used, the loss only needs to be calculated once
                if batch_i == 1
                    loss_jc, loss_cs, _, _ = protein_loss_grads("protein_$protein",
                                    training_sim_dir, traj_pes, traj_chem_shifts,
                                    frames_prot_val, false, models...)
                end
                if verbose
                    report("loss J-Coupling $loss_jc chemical shift $loss_cs\n")
                end
                if !isnan(loss_jc) && !isnan(loss_cs)
                    chunk_id = 1
                    push!(loss_J_coupling_chunks[chunk_id], loss_jc)
                    push!(loss_chem_shift_chunks[chunk_id], loss_cs)
                end
            end
        end

        losses_fs_intra      = vcat(loss_fs_intra_chunks...)
        losses_fs_inter      = vcat(loss_fs_inter_chunks...)
        losses_pe            = vcat(loss_pe_chunks...)
        losses_charges       = vcat(loss_charges_chunks...)
        losses_charge_reg    = vcat(loss_charge_reg_chunks...)
        losses_torsion_ks    = vcat(loss_torsion_ks_chunks...)
        losses_fs_intra_gems = vcat(loss_fs_intra_chunks_gems...)
        losses_fs_inter_gems = vcat(loss_fs_inter_chunks_gems...)
        losses_enth_vap      = vcat(loss_enth_vap_chunks...)
        losses_enth_mixing   = vcat(loss_enth_mixing_chunks...)
        losses_J_coupling    = vcat(loss_J_coupling_chunks...)
        losses_chem_shift    = vcat(loss_chem_shift_chunks...)
        if length(losses_fs_intra) > 0
            loss_sum_fs_intra_val      += sum(losses_fs_intra)
            loss_sum_fs_inter_val      += sum(losses_fs_inter)
            loss_sum_pe_val            += sum(losses_pe)
            loss_sum_charges_val       += sum(losses_charges)
            loss_sum_charge_reg_val    += sum(losses_charge_reg)
            loss_sum_torsion_ks_val    += sum(losses_torsion_ks)
            loss_sum_fs_intra_val_gems += sum(losses_fs_intra_gems)
            loss_sum_fs_inter_val_gems += sum(losses_fs_inter_gems)
            loss_sum_enth_vap_val      += sum(losses_enth_vap)
            loss_sum_enth_mixing_val   += sum(losses_enth_mixing)
            loss_sum_J_coupling_val    += sum(losses_J_coupling)
            loss_sum_chem_shift_val    += sum(losses_chem_shift)

            count_confs_val             += length(losses_fs_intra)
            count_confs_inter_val       += count(!iszero, losses_fs_inter)
            count_confs_pe_val          += length(losses_pe)
            count_confs_charges_val     += count(!iszero, losses_charges)
            count_confs_torsion_ks_val  += length(losses_torsion_ks)
            count_confs_val_gems        += length(losses_fs_intra_gems)
            count_confs_inter_val_gems  += count(!iszero, losses_fs_inter_gems)
            count_confs_enth_vap_val    += length(losses_enth_vap)
            count_confs_enth_mixing_val += length(losses_enth_mixing)
            count_confs_J_coupling_val  += length(losses_J_coupling)
        end
    end

    loss_mean_fs_intra_train      = loss_sum_fs_intra_train      / count_confs_train
    loss_mean_fs_inter_train      = loss_sum_fs_inter_train      / count_confs_inter_train
    loss_mean_pe_train            = loss_sum_pe_train            / count_confs_pe_train
    loss_mean_charges_train       = loss_sum_charges_train       / count_confs_charges_train
    loss_mean_charge_reg_train    = loss_sum_charge_reg_train    / count_confs_torsion_ks_train
    loss_mean_torsion_ks_train    = loss_sum_torsion_ks_train    / count_confs_torsion_ks_train
    loss_mean_fs_intra_train_gems = loss_sum_fs_intra_train_gems / count_confs_train_gems
    loss_mean_fs_inter_train_gems = loss_sum_fs_inter_train_gems / count_confs_inter_train_gems
    loss_mean_enth_vap_train      = loss_sum_enth_vap_train      / count_confs_enth_vap_train
    loss_mean_enth_mixing_train   = loss_sum_enth_mixing_train   / count_confs_enth_mixing_train
    loss_mean_J_coupling_train    = loss_sum_J_coupling_train    / count_confs_J_coupling_train
    loss_mean_chem_shift_train    = loss_sum_chem_shift_train    / count_confs_J_coupling_train

    loss_mean_fs_intra_val        = loss_sum_fs_intra_val        / count_confs_val
    loss_mean_fs_inter_val        = loss_sum_fs_inter_val        / count_confs_inter_val
    loss_mean_pe_val              = loss_sum_pe_val              / count_confs_pe_val
    loss_mean_charges_val         = loss_sum_charges_val         / count_confs_charges_val
    loss_mean_charge_reg_val      = loss_sum_charge_reg_val      / count_confs_torsion_ks_val
    loss_mean_torsion_ks_val      = loss_sum_torsion_ks_val      / count_confs_torsion_ks_val
    loss_mean_fs_intra_val_gems   = loss_sum_fs_intra_val_gems   / count_confs_val_gems
    loss_mean_fs_inter_val_gems   = loss_sum_fs_inter_val_gems   / count_confs_inter_val_gems
    loss_mean_enth_vap_val        = loss_sum_enth_vap_val        / count_confs_enth_vap_val
    loss_mean_enth_mixing_val     = loss_sum_enth_mixing_val     / count_confs_enth_mixing_val
    loss_mean_J_coupling_val      = loss_sum_J_coupling_val      / count_confs_J_coupling_val
    loss_mean_chem_shift_val      = loss_sum_chem_shift_val      / count_confs_J_coupling_val

    push!(epochs_mean_fs_intra_train     , loss_mean_fs_intra_train     )
    push!(epochs_mean_fs_intra_val       , loss_mean_fs_intra_val       )
    push!(epochs_mean_fs_inter_train     , loss_mean_fs_inter_train     )
    push!(epochs_mean_fs_inter_val       , loss_mean_fs_inter_val       )
    push!(epochs_mean_pe_train           , loss_mean_pe_train           )
    push!(epochs_mean_pe_val             , loss_mean_pe_val             )
    push!(epochs_mean_charges_train      , loss_mean_charges_train      )
    push!(epochs_mean_charges_val        , loss_mean_charges_val        )
    push!(epochs_mean_charge_reg_train   , loss_mean_charge_reg_train   )
    push!(epochs_mean_charge_reg_val     , loss_mean_charge_reg_val     )
    push!(epochs_mean_torsion_ks_train   , loss_mean_torsion_ks_train   )
    push!(epochs_mean_torsion_ks_val     , loss_mean_torsion_ks_val     )
    push!(epochs_mean_fs_intra_train_gems, loss_mean_fs_intra_train_gems)
    push!(epochs_mean_fs_intra_val_gems  , loss_mean_fs_intra_val_gems  )
    push!(epochs_mean_fs_inter_train_gems, loss_mean_fs_inter_train_gems)
    push!(epochs_mean_fs_inter_val_gems  , loss_mean_fs_inter_val_gems  )
    push!(epochs_mean_enth_vap_train     , loss_mean_enth_vap_train     )
    push!(epochs_mean_enth_vap_val       , loss_mean_enth_vap_val       )
    push!(epochs_mean_enth_mixing_train  , loss_mean_enth_mixing_train  )
    push!(epochs_mean_enth_mixing_val    , loss_mean_enth_mixing_val    )
    push!(epochs_mean_J_coupling_train   , loss_mean_J_coupling_train   )
    push!(epochs_mean_J_coupling_val     , loss_mean_J_coupling_val     )
    push!(epochs_mean_chem_shift_train   , loss_mean_chem_shift_train   )
    push!(epochs_mean_chem_shift_val     , loss_mean_chem_shift_val     )

    loss_regularisation = param_regularisation(models)
    push!(epochs_loss_regularisation, loss_regularisation)

    progress_str = ""
    if !isnothing(out_dir)
        for (store_id, default_str) in (
                ("val-val", "?"),
                ("ΔHvap"  , "ΔHvap water -, exp -, loss -"),
                ("ΔHmix"  , "ΔHmix CCCCO_OC1=NCCC1 - (- - -), exp -, loss -"),
            )
            store_fp = joinpath(out_dir, "store_$store_id.txt")
            if ispath(store_fp)
                progress_str *=  " - " * only(readlines(store_fp))
            else
                progress_str *=  " - " * default_str
            end
        end

        plot_training(
            joinpath(out_dir, "training.pdf"), models,
            epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
            epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
            epochs_mean_pe_train, epochs_mean_pe_val,
            epochs_mean_charges_train, epochs_mean_charges_val,
            epochs_mean_charge_reg_train, epochs_mean_charge_reg_val,
            epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
            epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
            epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems,
            epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
            epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
            epochs_mean_J_coupling_train, epochs_mean_J_coupling_val,
            epochs_mean_chem_shift_train, epochs_mean_chem_shift_val,
            epochs_loss_regularisation,
        )
        out_fp_models = joinpath(out_dir, "model.bson")
        out_fp_optims = joinpath(out_dir, "optim.bson")
        BSON.@save out_fp_models models
        BSON.@save out_fp_optims optims
        if save_every_epoch
            out_fp_models_epoch = joinpath(out_dir, "models", "model_ep_$epoch_n.bson")
            BSON.@save out_fp_models_epoch models
        end
    end

    time_epoch = now() - time_start
    time_wait_sims_perc = Int(round(100 * Dates.Second(round(time_wait_sims)) / time_epoch; digits=0))
    time_spice_perc     = Int(round(100 * Dates.Second(round(time_spice    )) / time_epoch; digits=0))
    time_gems_perc      = Int(round(100 * Dates.Second(round(time_gems     )) / time_epoch; digits=0))
    time_cond_perc      = Int(round(100 * Dates.Second(round(time_cond     )) / time_epoch; digits=0))
    time_protein_perc   = Int(round(100 * Dates.Second(round(time_protein  )) / time_epoch; digits=0))
    time_epoch_str = round(time_epoch, Minute)

    report("Epoch $epoch_n - mean training loss forces intra $loss_mean_fs_intra_train $loss_mean_fs_intra_train_gems force inter $loss_mean_fs_inter_train $loss_mean_fs_inter_train_gems pe $loss_mean_pe_train charge $loss_mean_charges_train charge reg $loss_mean_charge_reg_train torsion ks $loss_mean_torsion_ks_train ΔHvap $loss_mean_enth_vap_train ΔHmix $loss_mean_enth_mixing_train J-coupling $loss_mean_J_coupling_train chem shift $loss_mean_chem_shift_train regularisation $loss_regularisation - mean validation loss forces intra $loss_mean_fs_intra_val $loss_mean_fs_intra_val_gems force inter $loss_mean_fs_inter_val $loss_mean_fs_inter_val_gems pe $loss_mean_pe_val charge $loss_mean_charges_val charge reg $loss_mean_charge_reg_val torsion ks $loss_mean_torsion_ks_val ΔHvap $loss_mean_enth_vap_val ΔHmix $loss_mean_enth_mixing_val J-coupling $loss_mean_J_coupling_val chem shift $loss_mean_chem_shift_val$progress_str - $simulation_str - $time_spice_perc% SPICE, $time_gems_perc% GEMS, $time_cond_perc% condensed, $time_protein_perc% protein, $time_wait_sims_perc% sim waiting - took $time_epoch_str\n")

    GC.gc()
    return models, optims
end

function train!(models, optims)
    epochs_mean_fs_intra_train     , epochs_mean_fs_intra_val      = T[], T[]
    epochs_mean_fs_inter_train     , epochs_mean_fs_inter_val      = T[], T[]
    epochs_mean_pe_train           , epochs_mean_pe_val            = T[], T[]
    epochs_mean_charges_train      , epochs_mean_charges_val       = T[], T[]
    epochs_mean_charge_reg_train   , epochs_mean_charge_reg_val    = T[], T[]
    epochs_mean_torsion_ks_train   , epochs_mean_torsion_ks_val    = T[], T[]
    epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems = T[], T[]
    epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems = T[], T[]
    epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val      = T[], T[]
    epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val   = T[], T[]
    epochs_mean_J_coupling_train   , epochs_mean_J_coupling_val    = T[], T[]
    epochs_mean_chem_shift_train   , epochs_mean_chem_shift_val    = T[], T[]
    epochs_loss_regularisation = T[]

    if !isnothing(out_dir) && isfile(joinpath(out_dir, "training.log"))
        for line in readlines(joinpath(out_dir, "training.log"))
            if startswith(line, "Epoch")
                cols = split(line)
                push!(epochs_mean_fs_intra_train     , parse(T, cols[9 ]))
                push!(epochs_mean_fs_inter_train     , parse(T, cols[13]))
                push!(epochs_mean_pe_train           , parse(T, cols[16]))
                push!(epochs_mean_charges_train      , parse(T, cols[18]))
                push!(epochs_mean_charge_reg_train   , parse(T, cols[21]))
                push!(epochs_mean_torsion_ks_train   , parse(T, cols[24]))
                push!(epochs_mean_fs_intra_train_gems, parse(T, cols[10]))
                push!(epochs_mean_fs_inter_train_gems, parse(T, cols[14]))
                push!(epochs_mean_enth_vap_train     , parse(T, cols[26]))
                push!(epochs_mean_enth_mixing_train  , parse(T, cols[28]))
                push!(epochs_mean_J_coupling_train   , parse(T, cols[30]))
                push!(epochs_mean_chem_shift_train   , parse(T, cols[33]))
                push!(epochs_loss_regularisation     , parse(T, cols[35]))
                push!(epochs_mean_fs_intra_val       , parse(T, cols[42]))
                push!(epochs_mean_fs_inter_val       , parse(T, cols[46]))
                push!(epochs_mean_pe_val             , parse(T, cols[49]))
                push!(epochs_mean_charges_val        , parse(T, cols[51]))
                push!(epochs_mean_charge_reg_val     , parse(T, cols[54]))
                push!(epochs_mean_torsion_ks_val     , parse(T, cols[57]))
                push!(epochs_mean_fs_intra_val_gems  , parse(T, cols[43]))
                push!(epochs_mean_fs_inter_val_gems  , parse(T, cols[47]))
                push!(epochs_mean_enth_vap_val       , parse(T, cols[59]))
                push!(epochs_mean_enth_mixing_val    , parse(T, cols[61]))
                push!(epochs_mean_J_coupling_val     , parse(T, cols[63]))
                push!(epochs_mean_chem_shift_val     , parse(T, cols[66]))
            end
        end
        starting_epoch_n = length(epochs_mean_fs_intra_train) + 1
        trained_model = joinpath(out_dir, "model.bson")
        trained_optim = joinpath(out_dir, "optim.bson")
        BSON.@load trained_model models
        BSON.@load trained_optim optims
        report("Restarting training from epoch ", starting_epoch_n, " on ",
               Threads.nthreads(), " thread(s)\n")
    else
        starting_epoch_n = 1
        report("Starting training on ", Threads.nthreads(), " thread(s)\n")
    end

    for epoch_n in starting_epoch_n:n_epochs
        train_epoch!(
            models, optims, epoch_n,
            epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
            epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
            epochs_mean_pe_train, epochs_mean_pe_val,
            epochs_mean_charges_train, epochs_mean_charges_val,
            epochs_mean_charge_reg_train, epochs_mean_charge_reg_val,
            epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
            epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
            epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems,
            epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
            epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
            epochs_mean_J_coupling_train, epochs_mean_J_coupling_val,
            epochs_mean_chem_shift_train, epochs_mean_chem_shift_val,
            epochs_loss_regularisation,
        )
    end
    return models, optims
end

function evaluate_model(run_name, trained_model)
    # Make output directories first
    out_dir = "dft_eval"
    dist_nb_cutoff_val = T(1000.0)
    BSON.@load trained_model models
    Flux.testmode!(models)

    spicetest_hdf5_fp = "SPICE-test.hdf5"
    mol_features_spicetest = Dict(Pair(String.(split(line, "\t"; limit=2))...)
                                  for line in readlines("features_spicetest.tsv"))
    molecules_spicetest = keys(h5open(spicetest_hdf5_fp))
    molecules_com = vcat(molecules_test, molecules_spicetest)
    spicetest_flags = vcat(
        fill(false, length(molecules_test)),
        fill(true , length(molecules_spicetest)),
    )

    for (mol_id, spicetest_flag) in zip(molecules_com, spicetest_flags)
        any(s -> startswith(mol_id, s), ("maceoff_", "rna_")) && continue
        mol_id_clean = replace(mol_id, " " => "_")
        if spicetest_flag
            mol_hdf5 = h5read(spicetest_hdf5_fp, mol_id)
            feature_line = mol_features_spicetest[mol_id]
        else
            mol_hdf5 = h5read(spice_hdf5_fp, mol_id)
            feature_line = mol_features[mol_id]
        end

        if isdir(joinpath(out_dir, run_name, "energy", mol_id_clean))
            println(mol_id, " - already calculated")
            continue
        end

        n_confs = size(mol_hdf5["conformations"], 3)
        mkdir(joinpath(out_dir, run_name, "energy", mol_id_clean))
        mkdir(joinpath(out_dir, run_name, "forces", mol_id_clean))

        for conf_i in 1:n_confs
            coords = read_coordinates(mol_hdf5, conf_i)
            sys, _, _, _, _ = mol_to_system(mol_id, feature_line, coords, boundary_inf,
                                            models..., dist_nb_cutoff_val)
            neighbors = find_neighbors(sys)
            pe = potential_energy(sys, neighbors)
            fs = forces(sys, neighbors)

            open(joinpath(out_dir, run_name, "energy", mol_id_clean, "conf_$conf_i.txt"), "w") do of
                println(of, pe)
            end
            open(joinpath(out_dir, run_name, "forces", mol_id_clean, "conf_$conf_i.txt"), "w") do of
                for i in eachindex(fs)
                    println(of, fs[i][1], " ", fs[i][2], " ", fs[i][3])
                end
            end
        end

        println(mol_id, " - ", n_confs)
    end
end

function save_params(run_name, trained_model)
    # Make output directories first
    out_dir = "view_params"
    BSON.@load trained_model models
    Flux.testmode!(models)

    spicetest_hdf5_fp = "SPICE-test.hdf5"
    mol_features_spicetest = Dict(Pair(String.(split(line, "\t"; limit=2))...)
                                  for line in readlines("features_spicetest.tsv"))
    molecules_spicetest = keys(h5open(spicetest_hdf5_fp))
    element_σs = Dict(i => T[] for i in eachindex(element_i_to_name))
    element_ϵs = Dict(i => T[] for i in eachindex(element_i_to_name))

    for mol_id in molecules_spicetest
        mol_hdf5 = h5read(spicetest_hdf5_fp, mol_id)
        feature_line = mol_features_spicetest[mol_id]
        coords = read_coordinates(mol_hdf5, 1)
        sys, _, _, elements, _ = mol_to_system(mol_id, feature_line, coords, boundary_inf,
                                               models...)
        for (el, atom) in zip(elements, sys.atoms)
            push!(element_σs[el], atom.σ)
            push!(element_ϵs[el], atom.ϵ)
        end
        println(mol_id)
    end

    for el in eachindex(element_i_to_name)
        if length(element_σs[el]) > 0
            el_name = element_i_to_name[el]
            open(joinpath(out_dir, run_name, "sigmas_$el_name.txt"), "w") do of
                for x in element_σs[el]
                    println(of, x)
                end
            end
            open(joinpath(out_dir, run_name, "epsilons_$el_name.txt"), "w") do of
                for x in element_ϵs[el]
                    println(of, x)
                end
            end
        end
    end
end

if start_training
    train!(models, optims)
end
