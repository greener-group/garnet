import click
import pathlib
from garnetff import garnet 
from openff.pablo import topology_from_pdb
from openmm.app.pdbfile import PDBFile
from openff.toolkit import Molecule

# run this script with conda env garnet_rbfe

def write_xml_compatible_pdb(openff_topology, out_path):   
    """
    Converts an OpenFF topology to an OpenMM topology in which
    all residues of a protein are represented as a single
    residue. This is necessary because proteins are treated as 
    single residues in Garnet XML files.
    """
    
    # convert openff topology to openmm topology 
    openmm_top = openff_topology.to_openmm()                    
    openmm_pos = openff_topology.get_positions().to_openmm()
    
    chain_count = 1
    atom_count = 1
    
    for chain in openmm_top.chains(): 
        
        resi_list = []
        for resi in chain.residues():
            resi_list.append(resi)
    
        if len(resi_list) > 1:
            # rename protein residues so that all residues have the same name
            resi_name = f"M{chain_count}"
        else:
            # waters also need to be renamed, otherwise their CONECT won't be written out
            resi_name = f"{chain_count}" 
        
        for atom in chain.atoms():
            
            if atom_count <= 9999:                        # atom name field can't have more than four characters in PDB file
                atom.residue.name = resi_name             # set residue name (all residues of the protein should have the same name)
                atom.residue.id = f"{chain_count}"        # set residue number (all residues of the protein should have the same number)
                atom.name=f"{atom_count}"                 # set atom name (atom names must be unique when all residue names are the same)
                atom.residue.insertionCode = " "          # set insertion code
                atom_count+=1
            else:
                raise ValueError(("PDB file contains more than 9999 atoms, which is currently not supported."))
    
        chain_count+=1
    
    # write pdb file that matches the xml file 
    # must be one residue per polymer
    # must have CONECT (which can't automatically be inferred for non-standard residues) 
    PDBFile.writeFile(openmm_top, openmm_pos, file=out_path, keepIds=True, extraParticleIdentifier='EP') 


def add_molecules_to_topology(topology, ligand_mol_list, solvent_smiles_list): 
    """
    This function updates the input OpenFF topology by
    adding the ligand molecules specified in the a ligand
    sdf file as well as the (solvent) molecules found in 
    a list of SMILES. 
    """
    # add ligands to topology
    topology.add_molecules(ligand_mol_list)

    # create Molecule objects for other relevant system components
    solvent_mol_list = [Molecule.from_smiles(smiles_string, hydrogens_are_explicit=True) for smiles_string in solvent_smiles_list]

    # add wate and ions to topology
    topology.add_molecules(solvent_mol_list)

    return topology


@click.command
@click.option(
    '--pdb',
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to the prepared PDB file of the protein",
)
@click.option(
    '--ligands',
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to the prepared SDF file containing the ligands",
)
@click.option(
    '--cofactors',
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    help="Path to the prepared cofactors SDF file (optional)",
)
@click.option(
    '--output',
    type=click.Path(dir_okay=True, file_okay=False),
    required=True,
    help="Directory name in which to store the xml file and reformatted pdb file",
)

def run_inputs(pdb, ligands, output, cofactors=None):
    """
    Write XML and XML-compatible PDB files for input
    protein, ligand and cofactor structures. 

    Parameters
    ----------
    pdb : str
      Path to the protein PDB file.
    ligands : str
       Path to the SDF file containing ligand structures.
    output: str
      Path to the directory where output files will be written.
    cofactors : str, optional
      Path to an SDF file containing cofactor structures.
    """
    path = pathlib.Path(output)
    path.mkdir(parents=True, exist_ok=True)

    # read pdb file with openmm to solve common problems, like capping names 
    # (e.g. NMA instead of NME) and missing CONECT record
    # this makes it easier to read pdb file with openff later
    openmm_pdb = PDBFile(pdb)
    PDBFile.writeFile(
        openmm_pdb.topology, openmm_pdb.positions, file=f"{output}/protein_openmm.pdb", keepIds=False, extraParticleIdentifier='EP'
    )    
    pdb = f"{output}/protein_openmm.pdb" # update path
    
    # SMILES of molecules to add to topology
    # could also include other molecules if relevant
    solvent_smiles_list = ['[H]O[H]','[Na+]','[Cl-]'] 
    
    # create OpenFF topology from protein pdb file
    # must be OpenFF to be compatibale with topology_to_openmm_xml
    topology = topology_from_pdb(pdb)
    
    # write pdb file in format compatible with the xml file generated below
    write_xml_compatible_pdb(topology, f"{output}/protein.pdb")

    # read ligand sdf file
    ligand_mol_list = Molecule.from_file(ligands)

    # add cofactors 
    if cofactors is not None:
        cofactor_mol_list = Molecule.from_file(cofactors)
        if type(cofactor_mol_list) != list:
            cofactor_mol_list = [cofactor_mol_list]
        ligand_mol_list = ligand_mol_list + cofactor_mol_list 

    # update topology with relevant solvent and ligand molecules
    topology_all = add_molecules_to_topology(topology, ligand_mol_list, solvent_smiles_list)
    
    # run model and write xml file
    garnet.topology_to_openmm_xml(f"{output}/garnet.xml", topology_all)

if __name__ == "__main__":
    run_inputs()