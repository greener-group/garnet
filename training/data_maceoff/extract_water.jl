# Extract water clusters from MACE-OFF data

function extract()
    prev_line = ""
    atom_counter, conf_counter, n_atoms = 0, 0, 0
    writing = false
    of = nothing

    for line in Iterators.flatten((
                eachline("train_large_neut_no_bad_clean.xyz"),
                eachline("test_large_neut_all.xyz"),
            ))
        if writing
            atom_counter += 1
            println(of, line)
            if atom_counter == n_atoms
                writing = false
                close(of)
                @assert countlines("water/conf_$conf_counter.xyz") == n_atoms + 2
            end
        end
        if startswith(line, "Properties") && contains(line, "config_type=water")
            @assert !writing
            n_atoms = parse(Int, prev_line)
            atom_counter = 0
            conf_counter += 1
            of = open("water/conf_$conf_counter.xyz", "w")
            println(of, prev_line)
            println(of, line)
            writing = true
        end
        prev_line = line
    end
end

extract()
