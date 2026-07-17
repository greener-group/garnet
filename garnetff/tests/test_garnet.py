# Garnet tests

from garnetff import garnet
from openff.toolkit.topology import Molecule, Topology
from openff.pablo import topology_from_pdb
from openmm import LangevinMiddleIntegrator
from openmm.app import PME, HBonds, Simulation, PDBFile, PDBxFile
from openmm.unit import nanometer, picosecond, kelvin, kilojoules_per_mole
import os
import xml.etree.ElementTree as ET

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

def get_potential_energy(simulation):
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy()

def test_pdb():
    pdb_fp = os.path.join(data_dir, "gb3.pdb")
    topology = topology_from_pdb(pdb_fp)

    system, top_openmm = garnet.topology_to_openmm_system(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer,
        constraints=HBonds,
        rigidWater=True,
    )
    assert system.getNumParticles() == 9906

    temp = 300*kelvin
    dt = 0.004*picosecond
    integrator = LangevinMiddleIntegrator(temp, 1/picosecond, dt)
    simulation = Simulation(top_openmm, system, integrator)
    pdb = PDBFile(pdb_fp)
    simulation.context.setPositions(pdb.positions)
    assert get_potential_energy(simulation) < -100000 * kilojoules_per_mole

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temp)
    simulation.step(100)
    assert get_potential_energy(simulation) < -100000 * kilojoules_per_mole

def test_smiles():
    smiles = "[H]O[H]"
    mol = Molecule.from_smiles(smiles, hydrogens_are_explicit=True)
    topology = Topology.from_molecules(molecules=[mol])

    system, top_openmm = garnet.topology_to_openmm_system(topology)
    assert system.getNumParticles() == 3

def test_topology_xml_written(tmp_path):
    mol = Molecule.from_smiles("[H]O[H]", hydrogens_are_explicit=True)
    topology = Topology.from_molecules(molecules=[mol])
    ff_xml_fp = tmp_path / "water.xml"
    top_xml_fp = tmp_path / "water_residues.xml"

    garnet.topology_to_openmm_xml(
        ff_xml_fp,
        topology,
        mol_names=["HOH"],
        prefix="T",
        write_top=top_xml_fp,
    )

    assert ff_xml_fp.exists()
    assert top_xml_fp.exists()

    root = ET.parse(top_xml_fp).getroot()
    residues = root.findall("Residue")
    bonds = residues[0].findall("Bond")
    bond_names = {frozenset((bond.attrib["from"], bond.attrib["to"])) for bond in bonds}

    assert root.tag == "Residues"
    assert len(residues) == 1
    assert residues[0].attrib["name"] == "T_HOH"
    assert bond_names == {
        frozenset(("T_O1", "T_H1")),
        frozenset(("T_O1", "T_H2")),
    }

def test_pdb_write_requires_positions(tmp_path):
    mol = Molecule.from_smiles("[H]O[H]", hydrogens_are_explicit=True)
    topology = Topology.from_molecules(molecules=[mol])

    try:
        garnet.topology_to_openmm_xml(
            tmp_path / "water.xml",
            topology,
            write_pdb=tmp_path / "water.pdb",
        )
    except ValueError as e:
        assert str(e) == "write_pdb requires topology positions"
    else:
        assert False

def test_write_paths_do_not_accept_booleans(tmp_path):
    mol = Molecule.from_smiles("[H]O[H]", hydrogens_are_explicit=True)
    topology = Topology.from_molecules(molecules=[mol])

    for kw in ("write_top", "write_pdb"):
        try:
            garnet.topology_to_openmm_xml(tmp_path / f"{kw}.xml", topology, **{kw: True})
        except TypeError as e:
            assert str(e) == f"{kw} must be a file path or None"
        else:
            assert False

def test_pdb_written_with_xml_names(tmp_path):
    mol = Molecule.from_file(os.path.join(data_dir, "zw_l_alanine.sdf"))
    topology = Topology.from_molecules(molecules=[mol])
    ff_xml_fp = tmp_path / "alanine.xml"
    pdb_xml_fp = tmp_path / "alanine_named.pdb"

    garnet.topology_to_openmm_xml(ff_xml_fp, topology, mol_names=["M1"], write_pdb=pdb_xml_fp)

    assert ff_xml_fp.exists()
    assert pdb_xml_fp.exists()
    assert "CONECT" not in pdb_xml_fp.read_text()

    xml_root = ET.parse(ff_xml_fp).getroot()
    xml_residues = {
        residue.attrib["name"]: {atom.attrib["name"] for atom in residue.findall("Atom")}
        for residue in xml_root.find("Residues").findall("Residue")
    }
    pdb = PDBFile(str(pdb_xml_fp))

    for residue in pdb.topology.residues():
        assert residue.name in xml_residues
        for atom in residue.atoms():
            assert atom.name in xml_residues[residue.name]

def test_cif_written_for_long_residue_names(tmp_path):
    topology = topology_from_pdb(os.path.join(data_dir, "gb3.pdb"))
    ff_xml_fp = tmp_path / "gb3.xml"
    cif_fp = tmp_path / "gb3.cif"

    garnet.topology_to_openmm_xml(ff_xml_fp, topology, write_pdb=cif_fp)

    assert ff_xml_fp.exists()
    assert cif_fp.exists()
    assert sum(1 for _ in PDBxFile(str(cif_fp)).topology.residues()) == topology.n_molecules

def test_sdf():
    mol = Molecule.from_file(os.path.join(data_dir, "zw_l_alanine.sdf"))
    topology = Topology.from_molecules(molecules=[mol])

    system, top_openmm = garnet.topology_to_openmm_system(topology)
    assert system.getNumParticles() == 13
