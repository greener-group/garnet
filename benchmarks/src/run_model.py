# Get force field parameters with trained model
# Run from garnet conda environment

# Setup repo: pip install, readme with examples (ligand, NA, PTM, some from test), training script + files
# Pablo butanol fix: https://github.com/openforcefield/openff-pablo/issues/138
# Pablo PTM example
# Speed up isomorphism, special cases based on names?
# Multi xml sim fails: https://github.com/openmm/openmm/issues/5154
# Allow extra args passed to createSystem?
# Print info
# Indices as tensors/on GPU?
# Vectorise charge calculation
# Check dep versions, compare
# Device
# XML write header

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import SAGEConv
from openmm.app import ForceField, NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME
from openmm.unit import nanometer
import networkx as nx
from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from tempfile import NamedTemporaryFile
from warnings import warn

trained_model_fp = Path("/lmb/home/jgreener/dms/typing/") / "params_txt/64_7_ep12/model.pt"

element_indices = {
    1 : 0 , 3 : 1 , 5 : 2 , 6 : 3 , 7 : 4 , 8 : 5 , # H  Li B  C  N  O
    9 : 6 , 11: 7 , 12: 8 , 14: 9 , 15: 10, 16: 11, # F  Na Mg Si P  S
    17: 12, 19: 13, 20: 14, 35: 15, 53: 16,         # Cl K  Ca Br I
}

element_i_to_name = ["H", "Li", "B" , "C", "N" , "O" , "F", "Na", "Mg", "Si",
                     "P", "S" , "Cl", "K", "Ca", "Br", "I"]

atomic_masses = [
    1.008 , 6.94        , 10.81, 12.011, 14.007 , 15.999, 18.998403163, 22.98976928, 24.305,
    28.085, 30.973761998, 32.06, 35.45 , 39.0983, 40.078, 79.904      , 126.90447  ,
]

torsion_periodicities = list(range(1, 6 + 1))
torsion_phases = [0.0 if i % 2 == 0 else torch.pi for i in range(6)]

class DenseNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DenseNet, self).__init__()
        self.dense_1 = nn.Linear(dim_in, 128)
        self.dense_2 = nn.Linear(128, dim_out)

    def forward(self, input):
        h = F.relu(self.dense_1(input))
        o = self.dense_2(h)
        return o

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv_1 = SAGEConv(27, 128)
        self.conv_2 = SAGEConv(128, 64)

    def forward(self, data):
        h = F.relu(self.conv_1(data.x, data.edge_index))
        o = self.conv_2(h, data.edge_index)
        return o

def transform_σ(x):
    return torch.sigmoid(x) * 0.45 + 0.05 # 0.05 nm -> 0.5 nm

def transform_ϵ(x):
    return torch.sigmoid(x) * 1.48 + 0.02 # 0.02 kJ/mol -> 1.5 kJ/mol

def transform_bond_k(bf):
    return F.relu(bf.sum(dim=1))

def transform_angle_k(af):
    return F.relu(af.sum(dim=1))

def transform_bond_r0(bf):
    return F.relu((bf[:, 0] * 0.05 + bf[:, 1] * 0.3) / bf.sum(dim=1))

def transform_angle_θ0(af):
    return F.relu((af[:, 1] * torch.pi) / af.sum(dim=1)).clamp(0.0, torch.pi)

def find_identical_atoms(graph):
    n_atoms = len(graph)
    identical_atoms = [[] for _ in range(n_atoms)]
    for i in range(n_atoms):
        fci = graph.nodes[i]["formal_charge"]
        ei  = graph.nodes[i]["atomic_number"]
        if fci / fci.u != 0:
            identical_atoms[i].append(i)
            g1 = deepcopy(graph)
            g1.remove_node(i)
            for j in range(n_atoms):
                ej = graph.nodes[j]["atomic_number"]
                if i != j and ei == ej and nx.shortest_path_length(graph, i, j) < 5:
                    g2 = deepcopy(graph)
                    g2.remove_node(j)
                    iso = nx.vf2pp_isomorphism(g1, g2, node_label="atomic_number")
                    if iso is not None:
                        identical_atoms[i].append(j)
    return identical_atoms

class Model(nn.Module):
    def __init__(self, trained_model_fp=None):
        super(Model, self).__init__()
        self.atom_embed = GNN()
        self.bond_pool         = DenseNet(128, 64)
        self.angle_pool        = DenseNet(192, 64)
        self.proper_pool       = DenseNet(256, 64)
        self.improper_pool     = DenseNet(256, 64)
        self.atom_features     = DenseNet(64 , 4 )
        self.bond_features     = DenseNet(64 , 2 )
        self.angle_features    = DenseNet(64 , 2 )
        self.proper_features   = DenseNet(64 , 6 )
        self.improper_features = DenseNet(64 , 2 )

        self.weight_14_coul = 0.0
        self.dexp_alpha     = 0.0
        self.dexp_beta      = 0.0

        if trained_model_fp is not None:
            loaded_dict = torch.load(trained_model_fp, weights_only=True)

            self.atom_embed.load_state_dict(loaded_dict["atom_embed"])
            self.bond_pool.load_state_dict(loaded_dict["bond_pool"])
            self.angle_pool.load_state_dict(loaded_dict["angle_pool"])
            self.proper_pool.load_state_dict(loaded_dict["proper_pool"])
            self.improper_pool.load_state_dict(loaded_dict["improper_pool"])
            self.atom_features.load_state_dict(loaded_dict["atom_features"])
            self.bond_features.load_state_dict(loaded_dict["bond_features"])
            self.angle_features.load_state_dict(loaded_dict["angle_features"])
            self.proper_features.load_state_dict(loaded_dict["proper_features"])
            self.improper_features.load_state_dict(loaded_dict["improper_features"])

            self.weight_14_coul = loaded_dict["weight_14_coul"]
            self.dexp_alpha     = loaded_dict["dexp_alpha"]
            self.dexp_beta      = loaded_dict["dexp_beta"]

    def topology_to_data(self, topology, mol_names=None):
        if topology.n_atoms == 0:
            raise ValueError("topology doesn't contain any atoms")
        if mol_names is not None and len(mol_names) != topology.n_molecules:
            raise ValueError((f"The length of mol_names ({len(mol_names)}) must be the same as "
                              f"the number of molecules in the topology ({topology.n_molecules})"))

        atom_features, elements, molecule_inds, formal_charges = [], [], [], []
        edge_is, edge_js = [], []
        bond_is, bond_js = [], []
        angle_is, angle_js, angle_ks = [], [], []
        proper_is, proper_js, proper_ks, proper_ls = [], [], [], []
        improper_is, improper_js, improper_ks, improper_ls = [], [], [], []
        mol_names_used = []
        prev_atom_count, prev_mol_count = 0, 0
        mols = list(topology.molecules)

        for mol_i, mol in enumerate(mols):
            if mol.n_atoms == 0:
                continue
            already_processed = False
            for mol_j in range(mol_i):
                if mol.is_isomorphic_with(mols[mol_j]):
                    already_processed = True
                    break
            if already_processed:
                continue

            for i, a in enumerate(mol.atoms):
                an = a.atomic_number
                if an not in element_indices:
                    raise ValueError((f"Atom {i+1} in molecule {mol_i+1} has atomic number {an} "
                                      "but this element is not supported"))
                elements.append(element_indices[an])

            fcs = []
            for i, a in enumerate(mol.atoms):
                c = int(a.formal_charge / a.formal_charge.units)
                if c not in (-1, 0, 1, 2):
                    raise ValueError((f"Atom {i+1} in molecule {mol_i+1} has charge {c} but only "
                                      "charges in (-1, 0, 1, 2) are supported"))
                fcs.append(c)
            formal_charges.extend(fcs)

            graph = mol.to_networkx()
            edge_is.extend(range(prev_atom_count, prev_atom_count + mol.n_atoms))
            edge_js.extend(range(prev_atom_count, prev_atom_count + mol.n_atoms))

            identical_atoms = find_identical_atoms(graph)
            spread_charges = [0.0] * mol.n_atoms
            for i in range(mol.n_atoms):
                if len(identical_atoms[i]) > 0:
                    for j in identical_atoms[i]:
                        spread_charges[j] += fcs[i] / len(identical_atoms[i])

            for i, a in enumerate(mol.atoms):
                at_feat = [0.0] * 27
                at_feat[elements[prev_atom_count + i]] = 1.0
                at_feat[17] =  spread_charges[i] / 2 if spread_charges[i] > 0 else 0.0
                at_feat[18] = -spread_charges[i] / 2 if spread_charges[i] < 0 else 0.0
                at_feat[19] = 1.0 if a.is_aromatic else 0.0
                at_feat[20 + min(len(list(a.bonded_atoms)), 6)] = 1.0
                atom_features.append(at_feat)

            if len(mol.bonds) > 0:
                bond_indices = [(b.atom1_index + prev_atom_count,
                                 b.atom2_index + prev_atom_count) for b in mol.bonds]
                bond_indices = [(i, j) if i < j else (j, i) for i, j in bond_indices]
                sorted_inds = sorted(bond_indices, key=itemgetter(0, 1))
                si_is = [si[0] for si in sorted_inds]
                si_js = [si[1] for si in sorted_inds]
                bond_is.extend(si_is)
                bond_js.extend(si_js)
                edge_is.extend(si_is)
                edge_js.extend(si_js)
                edge_is.extend(si_js)
                edge_js.extend(si_is)

            if len(mol.angles) > 0:
                angle_indices = [(a[0].molecule_atom_index + prev_atom_count,
                                  a[1].molecule_atom_index + prev_atom_count,
                                  a[2].molecule_atom_index + prev_atom_count) for a in mol.angles]
                angle_indices = [(i, j, k) if i < k else (k, j, i) for i, j, k in angle_indices]
                sorted_inds = sorted(angle_indices, key=itemgetter(0, 1, 2))
                angle_is.extend([si[0] for si in sorted_inds])
                angle_js.extend([si[1] for si in sorted_inds])
                angle_ks.extend([si[2] for si in sorted_inds])

            if len(mol.propers) > 0:
                proper_indices = [(p[0].molecule_atom_index + prev_atom_count,
                                   p[1].molecule_atom_index + prev_atom_count,
                                   p[2].molecule_atom_index + prev_atom_count,
                                   p[3].molecule_atom_index + prev_atom_count) for p in mol.propers]
                proper_indices = [(i, j, k, l) if i < l else (l, k, j, i) for i, j, k, l in proper_indices]
                sorted_inds = sorted(proper_indices, key=itemgetter(0, 1, 2, 3))
                proper_is.extend([si[0] for si in sorted_inds])
                proper_js.extend([si[1] for si in sorted_inds])
                proper_ks.extend([si[2] for si in sorted_inds])
                proper_ls.extend([si[3] for si in sorted_inds])

            if len(mol.amber_impropers) > 0:
                improper_indices = []
                for improper in mol.amber_impropers:
                    central_i = improper[0].molecule_atom_index
                    other_is = sorted(a.molecule_atom_index for a in improper[1:])
                    inds = [i + prev_atom_count for i in [central_i, *other_is]]
                    if inds not in improper_indices:
                        improper_indices.append(inds)
                sorted_inds = sorted(improper_indices, key=itemgetter(0, 1, 2, 3))
                improper_is.extend([si[0] for si in sorted_inds])
                improper_js.extend([si[1] for si in sorted_inds])
                improper_ks.extend([si[2] for si in sorted_inds])
                improper_ls.extend([si[3] for si in sorted_inds])

            mis = [-1] * mol.n_atoms
            for ci, cc in enumerate(nx.connected_components(graph)):
                for ai in cc:
                    mis[ai] = prev_mol_count + ci
            assert -1 not in mis
            molecule_inds.extend(mis)
            name_prefix = f"MOL{mol_i+1}" if mol_names is None else mol_names[mol_i]
            n_mols = len(set(mis))
            if n_mols == 1:
                mol_names_used.append(name_prefix)
            else:
                for mi in range(n_mols):
                    mol_names_used.append(f"{name_prefix}_{mi+1}")

            prev_atom_count += mol.n_atoms
            prev_mol_count = max(molecule_inds) + 1

        assert len(mol_names_used) == max(molecule_inds) + 1
        x = torch.tensor(atom_features)
        edge_index = torch.tensor([edge_is, edge_js], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, elements=elements,
                    formal_charges=formal_charges, bond_is=bond_is, bond_js=bond_js,
                    angle_is=angle_is, angle_js=angle_js, angle_ks=angle_ks,
                    proper_is=proper_is, proper_js=proper_js, proper_ks=proper_ks, proper_ls=proper_ls,
                    improper_is=improper_is, improper_js=improper_js, improper_ks=improper_ks,
                    improper_ls=improper_ls, molecule_inds=molecule_inds, mol_names=mol_names_used)
        return data

    def data_to_xml(self, ff_xml_fp, data, prefix=None):

        atom_embeddings = self.atom_embed(data)               
        atom_features = self.atom_features(atom_embeddings)   

        bond_emb_1     = atom_embeddings[data.bond_is    , :] 
        bond_emb_2     = atom_embeddings[data.bond_js    , :] 
        angle_emb_1    = atom_embeddings[data.angle_is   , :] 
        angle_emb_2    = atom_embeddings[data.angle_js   , :] 
        angle_emb_3    = atom_embeddings[data.angle_ks   , :] 
        proper_emb_1   = atom_embeddings[data.proper_is  , :] 
        proper_emb_2   = atom_embeddings[data.proper_js  , :] 
        proper_emb_3   = atom_embeddings[data.proper_ks  , :] 
        proper_emb_4   = atom_embeddings[data.proper_ls  , :] 
        improper_emb_1 = atom_embeddings[data.improper_is, :] 
        improper_emb_2 = atom_embeddings[data.improper_js, :] 
        improper_emb_3 = atom_embeddings[data.improper_ks, :] 
        improper_emb_4 = atom_embeddings[data.improper_ls, :] 

        bond_com_emb_1 = torch.cat((bond_emb_1, bond_emb_2), dim=1) 
        bond_com_emb_2 = torch.cat((bond_emb_2, bond_emb_1), dim=1) 
        bond_pool_1 = self.bond_pool(bond_com_emb_1)                
        bond_pool_2 = self.bond_pool(bond_com_emb_2)                
        bond_embeddings = bond_pool_1 + bond_pool_2                 

        angle_com_emb_1 = torch.cat((angle_emb_1, angle_emb_2, angle_emb_3), dim=1) 
        angle_com_emb_2 = torch.cat((angle_emb_3, angle_emb_2, angle_emb_1), dim=1) 
        angle_pool_1 = self.angle_pool(angle_com_emb_1)                             
        angle_pool_2 = self.angle_pool(angle_com_emb_2)                             
        angle_embeddings = angle_pool_1 + angle_pool_2                              

        proper_com_emb_1 = torch.cat((proper_emb_1, proper_emb_2, proper_emb_3, proper_emb_4), dim=1) 
        proper_com_emb_2 = torch.cat((proper_emb_4, proper_emb_3, proper_emb_2, proper_emb_1), dim=1) 
        proper_pool_1 = self.proper_pool(proper_com_emb_1)                                            
        proper_pool_2 = self.proper_pool(proper_com_emb_2)                                            
        proper_embeddings = proper_pool_1 + proper_pool_2                                             

        improper_com_emb_1 = torch.cat((improper_emb_1, improper_emb_2, improper_emb_3, improper_emb_4), dim=1) 
        improper_com_emb_2 = torch.cat((improper_emb_1, improper_emb_3, improper_emb_2, improper_emb_4), dim=1) 
        improper_com_emb_3 = torch.cat((improper_emb_1, improper_emb_4, improper_emb_2, improper_emb_3), dim=1) 
        improper_pool_1 = self.improper_pool(improper_com_emb_1)                                                
        improper_pool_2 = self.improper_pool(improper_com_emb_2)                                                
        improper_pool_3 = self.improper_pool(improper_com_emb_3)                                                
        improper_embeddings = improper_pool_1 + improper_pool_2 + improper_pool_3                               

        bond_features     = self.bond_features(bond_embeddings)         
        angle_features    = self.angle_features(angle_embeddings)       
        proper_features   = self.proper_features(proper_embeddings)     
        improper_features = self.improper_features(improper_embeddings) 

        charge_e = atom_features[:, 0]
        charge_inv_s = 1 / atom_features[:, 1]
        charge_e_inv_s = charge_e * charge_inv_s

        n_molecules = int(max(data.molecule_inds)) + 1
        mol_charge_factors = [0.0] * n_molecules

        for mi in range(n_molecules):
            mol_charge, mol_charge_e_inv_s, mol_charge_inv_s = 0.0, 0.0, 0.0
            for (ai, ami) in enumerate(data.molecule_inds):
                if ami == mi:
                    mol_charge += data.formal_charges[ai]
                    mol_charge_e_inv_s += charge_e_inv_s[ai]
                    mol_charge_inv_s += charge_inv_s[ai]
            mol_charge_factors[mi] = (mol_charge + mol_charge_e_inv_s) / mol_charge_inv_s

        # Reverted back to implicit CPU tensor
        charge_factors = torch.tensor([mol_charge_factors[mi] for mi in data.molecule_inds])
        partial_charges = -charge_e_inv_s + charge_inv_s * charge_factors

        σs = transform_σ(atom_features[:, 2])
        ϵs = transform_ϵ(atom_features[:, 3])
        bond_fcs = transform_bond_k(bond_features)
        bond_r0s = transform_bond_r0(bond_features)
        angle_fcs = transform_angle_k(angle_features)
        angle_θ0s = transform_angle_θ0(angle_features)

        prefix_str = "" if prefix is None else prefix + "_"
        element_counts = [0] * len(element_i_to_name)
        atom_types = []
        for ei in data.elements:
            element_counts[ei] += 1
            el = element_i_to_name[ei]
            atom_types.append(f"{prefix_str}{el}{element_counts[ei]}")

        with open(ff_xml_fp, "wt") as of:
            of.write("<ForceField>\n")
            of.write("  <AtomTypes>\n")
            for (at, ei) in zip(atom_types, data.elements):
                el, am = element_i_to_name[ei], atomic_masses[ei]
                of.write(f"    <Type element=\"{el}\" name=\"{at}\" class=\"{at}\" mass=\"{am}\"/>\n")
            of.write("  </AtomTypes>\n")

            of.write("  <Residues>\n")
            for mol_i in range(n_molecules):
                mol_name = prefix_str + data.mol_names[mol_i]
                atom_inds = [i for i, mi in enumerate(data.molecule_inds) if mi == mol_i]
                of.write(f"    <Residue name=\"{mol_name}\">\n")
                for ai in atom_inds:
                    at, pc = atom_types[ai], partial_charges[ai]
                    of.write(f"      <Atom name=\"{at}\" type=\"{at}\" charge=\"{pc}\"/>\n")
                for bi, bj in zip(data.bond_is, data.bond_js):
                    if bi in atom_inds and bj in atom_inds:
                        at1, at2 = atom_types[bi], atom_types[bj]
                        of.write(f"      <Bond atomName1=\"{at1}\" atomName2=\"{at2}\"/>\n")
                    elif bi in atom_inds or bj in atom_inds:
                        raise ValueError(f"Atoms {bi+1} and {bj+1} are bonded but are in different molecules")
                of.write("    </Residue>\n")
            of.write("  </Residues>\n")

            if len(data.bond_is) > 0:
                of.write("  <HarmonicBondForce>\n")
                for (ai, aj, k, r0) in zip(data.bond_is, data.bond_js, bond_fcs, bond_r0s):
                    at1, at2 = atom_types[ai], atom_types[aj]
                    of.write(f"    <Bond type1=\"{at1}\" type2=\"{at2}\" length=\"{r0}\" k=\"{k}\"/>\n")
                of.write("  </HarmonicBondForce>\n")

            if len(data.angle_is) > 0:
                of.write("  <HarmonicAngleForce>\n")
                for (ai, aj, ak, k, θ0) in zip(data.angle_is, data.angle_js, data.angle_ks,
                                               angle_fcs, angle_θ0s):
                    at1, at2, at3 = atom_types[ai], atom_types[aj], atom_types[ak]
                    of.write((f"    <Angle type1=\"{at1}\" type2=\"{at2}\" type3=\"{at3}\" "
                              f"angle=\"{θ0}\" k=\"{k}\"/>\n"))
                of.write("  </HarmonicAngleForce>\n")

            if len(data.proper_is) > 0 or len(data.improper_is) > 0:
                of.write("  <PeriodicTorsionForce ordering=\"default\">\n")
                for prop_i, (ai, aj, ak, al) in enumerate(zip(data.proper_is, data.proper_js,
                                                              data.proper_ks, data.proper_ls)):
                    at1, at2, at3, at4 = atom_types[ai], atom_types[aj], atom_types[ak], atom_types[al]
                    s = f"    <Proper type1=\"{at1}\" type2=\"{at2}\" type3=\"{at3}\" type4=\"{at4}\""
                    for ti in range(6):
                        tn = ti + 1
                        tpe, tph = torsion_periodicities[ti], torsion_phases[ti]
                        k = proper_features[prop_i, ti]
                        s += f" periodicity{tn}=\"{tpe}\" phase{tn}=\"{tph}\" k{tn}=\"{k}\""
                    of.write(s + "/>\n")

                for improp_i, (ai, aj, ak, al) in enumerate(zip(data.improper_is, data.improper_js,
                                                                data.improper_ks, data.improper_ls)):
                    at1, at2, at3, at4 = atom_types[ai], atom_types[aj], atom_types[ak], atom_types[al]
                    s = f"    <Improper type1=\"{at1}\" type2=\"{at2}\" type3=\"{at3}\" type4=\"{at4}\""
                    for ti in range(2):
                        tn = ti + 1
                        tpe, tph = torsion_periodicities[ti], torsion_phases[ti]
                        k = improper_features[improp_i, ti]
                        s += f" periodicity{tn}=\"{tpe}\" phase{tn}=\"{tph}\" k{tn}=\"{k}\""
                    of.write(s + "/>\n")
                of.write("  </PeriodicTorsionForce>\n")

            of.write(f"  <NonbondedForce coulomb14scale=\"{self.weight_14_coul}\" lj14scale=\"0.0\">\n")
            of.write("    <UseAttributeFromResidue name=\"charge\"/>\n")
            for at in atom_types:
                of.write(f"    <Atom type=\"{at}\" sigma=\"1\" epsilon=\"0\"/>\n")
            of.write("  </NonbondedForce>\n")

            of.write(("  <CustomNonbondedForce energy=\"sqrt(epsilon1*epsilon2)*"
                      "(((beta*exp(alpha))/(alpha-beta))*exp(-alpha*(r/((2^(1/6))*((sigma1+sigma2)/2))))"
                      "-((alpha*exp(beta))/(alpha-beta))*exp(-beta*(r/((2^(1/6))*((sigma1+sigma2)/2))))"
                      ")\" bondCutoff=\"3\">\n"))
            of.write(f"    <GlobalParameter name=\"alpha\" defaultValue=\"{self.dexp_alpha}\"/>\n")
            of.write(f"    <GlobalParameter name=\"beta\" defaultValue=\"{self.dexp_beta}\"/>\n")
            of.write("    <PerParticleParameter name=\"sigma\"/>\n")
            of.write("    <PerParticleParameter name=\"epsilon\"/>\n")
            for at, σ, ϵ in zip(atom_types, σs, ϵs):
                of.write(f"    <Atom type=\"{at}\" sigma=\"{σ}\" epsilon=\"{ϵ}\"/>\n")
            of.write("  </CustomNonbondedForce>\n")
            of.write("</ForceField>\n")

    def topology_to_openmm_xml(self, ff_xml_fp, topology, mol_names=None, prefix=None):
        with torch.no_grad():
            # Reverted back to CPU execution
            data = self.topology_to_data(topology, mol_names=mol_names)
            self.data_to_xml(ff_xml_fp, data, prefix=prefix)

    def topology_to_openmm_system(self, topology, nonbondedMethod=NoCutoff,
                                  nonbondedCutoff=1*nanometer, constraints=None, rigidWater=False, removeCMMotion=True, hydrogenMass=None, switchDistance=None,
                                  useDispersionCorrection=False):
        
        if nonbondedMethod in (NoCutoff, CutoffNonPeriodic):
            if topology.box_vectors is not None:
                warn((f"nonbondedMethod is {nonbondedMethod} but the topology has box "
                      "vectors set, this may indicate an issue"))
        elif nonbondedMethod in (CutoffPeriodic, Ewald, PME, LJPME):
            pass
        else:
            raise ValueError(("nonbondedMethod must be one of NoCutoff, CutoffNonPeriodic, "
                              "CutoffPeriodic, Ewald, PME or LJPME"))
        
        mols = list(topology.molecules)
        polymer_count = 0
        for mol_i, mol in enumerate(mols):
            smiles = mol.to_smiles()
            if "." in smiles:
                raise ValueError((f"Molecule {mol_i+1} with SMILES {smiles} contains more than one "
                                  "molecule, this is not supported for topology_to_openmm_system"))
            res_names = []
            for a in mol.atoms:
                resnum  = a.metadata["residue_number"] if "residue_number" in a.metadata else " "
                inscode = a.metadata["insertion_code"] if "insertion_code" in a.metadata else " "
                res_names.append(f"{resnum}_{inscode}")
            if len(set(res_names)) > 1:
                polymer_count += 1
                res_num = mol.atoms[0].metadata["residue_number"]
                for a in mol.atoms:
                    a.metadata["residue_name"] = f"M{polymer_count}"
                    a.metadata["residue_number"] = res_num
                    a.metadata["insertion_code"] = " "

        top_openmm = topology.to_openmm()
        with torch.no_grad():
            # Reverted back to CPU execution
            data = self.topology_to_data(topology)

            with NamedTemporaryFile(delete=False) as temp_file:
                temp_fp = temp_file.name
                self.data_to_xml(temp_fp, data)
                
        ff = ForceField(temp_fp)
        system = ff.createSystem(
            top_openmm,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff, 
            constraints=constraints,
            rigidWater=rigidWater,
            removeCMMotion=removeCMMotion,
            hydrogenMass=hydrogenMass,
            switchDistance=switchDistance,
            useDispersionCorrection=useDispersionCorrection,
        )
        return system, top_openmm

garnet = Model(trained_model_fp)