# Generate features for MACE-OFF water data training
# Assumes all formal charges are zero, since charges are not spread to equivalent atoms

out_fp = "features_maceoff.tsv"
data_dir = "../data_maceoff/water"

n_confs = length(readdir(data_dir))

open(out_fp, "w") do of
    for conf_i in 1:n_confs
        n_molecules = parse(Int, readlines("$data_dir/conf_$conf_i.xyz")[1]) ÷ 3

        bonds, angles = String[], String[]
        for Oi in 1:n_molecules
            H1i = n_molecules + 2*Oi - 1
            H2i = n_molecules + 2*Oi
            push!(bonds, "$Oi/$H1i")
            push!(bonds, "$Oi/$H2i")
            push!(angles, "$H1i/$Oi/$H2i")
        end

        mol_id = "maceoff_water_$conf_i"
        str_element       = join(vcat(fill(6, n_molecules), fill(1, 2*n_molecules)), ",")
        str_charge        = join(fill(0.0, 3*n_molecules), ",")
        str_aromatic      = join(fill(0, 3*n_molecules), ",")
        str_nbonded       = join(vcat(fill(2, n_molecules), fill(1, 2*n_molecules)), ",")
        str_bond          = join(bonds, ",")
        str_angle         = join(angles, ",")
        str_proper        = "-"
        str_improper      = "-"
        str_molecule_inds = join(vcat(1:n_molecules, repeat(1:n_molecules; inner=2)), ",")

        println(of, join(
            [
                mol_id, str_element, str_charge, str_aromatic, str_nbonded, str_bond,
                str_angle, str_proper, str_improper, str_molecule_inds,
            ],
            "\t",
        ))
    end
end
