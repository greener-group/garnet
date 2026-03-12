import numpy as np

karplus_dict = {("HA", "CA", "N",  "C")   : {"ALL" : ( 120.0, 3.72, -2.18,  1.28, 0.38)},
                ("HN", "N",  "CA", "CB")  : {"ALL" : (  60.0, 3.51, -0.53,  0.14, 0.25)},
                ("HN", "N",  "CA", "C")   : {"ALL" : ( 180.0, 4.12, -1.10,  0.11, 0.31)},
                ("HN", "N",  "CA", "HA")  : {"ALL" : ( -60.0, 7.97, -1.26,  0.63, 0.42)},
                ("C",  "CA", "CB", "CG1") : {"VAL" : (-115.0, 3.42, -0.59,  0.17, 0.25)},
                ("C",  "CA", "CB", "CG2") : {"VAL" : (   5.0, 3.42, -0.59,  0.17, 0.25),
                                             "ILE" : ( 125.0, 3.42, -0.59,  0.17, 0.25),
                                             "THR" : ( 137.0, 2.76, -0.67,  0.19, 0.21)},
                ("N",  "CA", "CB", "CG1") : {"VAL" : (   6.0, 2.64,  0.26, -0.22, 0.25)},
                ("N",  "CA", "CB", "CG2") : {"VAL" : ( 126.0, 2.64,  0.26, -0.22, 0.25),
                                             "ILE" : (-114.0, 2.64,  0.26, -0.22, 0.25),
                                             "THR" : (-113.0, 2.01,  0.21, -0.12, 0.21)},
                ("HA", "CA", "CB", "HB")  : {"VAL" : (   0.0, 7.23, -1.37,  1.79, 0.40),
                                             "ILE" : (   0.0, 7.23, -1.37,  1.79, 0.40),
                                             "THR" : (   0.0, 7.23, -1.37,  0.81, 0.40)},
                ("HA", "CA", "CB", "HB2") : {"ALL" : (-120.0, 7.23, -1.37,  2.40, 0.40),
                                             "CYS" : (-120.0, 7.23, -1.37,  1.71, 0.40),
                                             "SER" : (-120.0, 7.23, -1.37,  1.42, 0.40)},
                ("HA", "CA", "CB", "HB3") : {"ALL" : (   0.0, 7.23, -1.37,  2.40, 0.40),
                                             "CYS" : (   0.0, 7.23, -1.37,  1.71, 0.40),
                                             "SER" : (   0.0, 7.23, -1.37,  1.42, 0.40)}}

backbone_torsions = [("HA", "CA", "N",  "C"),
                     ("HN", "N",  "CA", "CB"),
                     ("HN", "N",  "CA", "C"),
                     ("HN", "N",  "CA", "HA")]

def karplus_J(theta, params):

    delta, A, B, C, sigma = params
    delta = np.deg2rad(delta)

    return A*np.cos(theta + delta)**2 + B*np.cos(theta + delta) + C


def karplus_extrema(params):
    """
    Return min and max of the Karplus function
    A*cos^2(phi) + B*cos(phi) + C, phi arbitrary.
    The delta shift does not affect the extrema.
    """
    delta, A, B, C, sigma = params

    # Evaluate f(x) = A x^2 + B x + C on [-1, 1]
    # Candidates: x = -1, x = 1, and the vertex x = -B/(2A) if inside [-1,1]
    xs = [-1.0, 1.0]

    if A != 0:
        x_crit = -B / (2*A)
        if -1.0 <= x_crit <= 1.0:
            xs.append(x_crit)

    vals = [A*x*x + B*x + C for x in xs]
    return min(vals), max(vals)

def karplus_hbonds(R, theta, phi):
    
    k  =  3.2  # Angstroms
    R0 =  1.76 # Angstroms
    A  =  0.62 # Hz
    B  =  0.92 # Hz
    C  =  0.14 # Hz
    D  = -1.31 # HZ

    return np.exp(-k*(R - R0)) * ((A * np.cos(phi)**2 + B * np.cos(phi) + C) * np.sin(theta) ** 2 + D * np.cos(theta)**2)

# ---------- basic geometry helpers ----------

def angle(a, b, c):
    """Return angle ABC (in radians) given 3 points a, b, c as (3,) arrays."""
    ba = a - b
    bc = c - b
    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)
    cosang = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return np.arccos(cosang)


def dihedral(p1, p2, p3, p4):
    """Return dihedral angle (in radians) for four points p1-p2-p3-p4."""
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3

    # Normalize b1 so that it does not affect magnitude of vector
    b1 /= np.linalg.norm(b1)

    # Compute normals
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


# ---------- hydrogen bond + geometry ----------

def hbond_geometry(N_d, H_d, O_a, C_a, N_a, max_NO=3.5, min_angle=2.094):
    """
    Check N_d–H_d…O_a=C_a–N_a backbone H-bond and, if present, return geometry.

    Returns
    -------
    (is_hbond, R_HO, theta_HOC, phi_HOCN)
      is_hbond : bool
      R_HO     : float | None, distance H…O (Å)
      theta_HOC: float | None, angle H–O–C (deg)
      phi_HOCN : float | None, dihedral H–O–C–N (deg)
    """
    if (len(N_d) == 0 or len(H_d) == 0 or len(O_a) == 0 or
        len(C_a) == 0 or len(N_a) == 0):
        return False, None, None, None

    N_d_pos = N_d.positions[0]
    H_d_pos = H_d.positions[0]
    O_a_pos = O_a.positions[0]
    C_a_pos = C_a.positions[0]
    N_a_pos = N_a.positions[0]

    # Distance-based screening
    NO_dist = np.linalg.norm(N_d_pos - O_a_pos)
    if NO_dist > max_NO:
        return False, None, None, None

    HO_dist = np.linalg.norm(H_d_pos - O_a_pos)

    # Angle at H in the usual H-bond sense N–H…O
    ang_NHO = angle(N_d_pos, H_d_pos, O_a_pos)
    if ang_NHO < min_angle:
        return False, None, None, None

    # If we are here, we consider this a hydrogen bond.
    # Now compute requested geometry:
    #  - R: H…O distance
    #  - theta: H–O–C
    #  - phi: H–O–C–N
    R_HO      = HO_dist
    theta_HOC = angle(H_d_pos, O_a_pos, C_a_pos)
    phi_HOCN  = dihedral(H_d_pos, O_a_pos, C_a_pos, N_a_pos)

    return True, R_HO, theta_HOC, phi_HOCN


import numpy as np

def backbone_amide_hbond_between(res_i, res_j, max_NO=3.5, min_angle=2.094):
    """
    Check for a backbone amide H-bond between two residues.

    We look for a backbone N–H ... O=C hydrogen bond either with
    i as donor and j as acceptor ("i->j") or j as donor and i as
    acceptor ("j->i").

    Returns
    -------
    is_hb : bool
        True if an H-bond was found in either direction.
    direction : {"i->j", "j->i", None}
        Direction of the H-bond if found, otherwise None.
    R_HO : float or None
        H...O distance (Å) used in the Barfield expression.
    theta_HOC : float or None
        H–O–C angle (radians).
    phi_HOCN : float or None
        H–O–C–N dihedral (radians).

    Notes
    -----
    For the Barfield 3J_{N,C'} expression, the "acceptor nitrogen" is the
    amide nitrogen covalently bonded to the carbonyl C' that acts as
    acceptor. In a peptide, if C' belongs to residue j, that nitrogen is
    in residue j+1 (the next residue along the same chain), not N_j.
    """

    u = res_i.universe

    # --- helper: get backbone N, H, O, C AtomGroups for a residue ---
    def backbone_groups(res):
        N = res.atoms.select_atoms("name N")
        H = res.atoms.select_atoms("name H or name HN")
        O = res.atoms.select_atoms("name O")
        C = res.atoms.select_atoms("name C")
        return N, H, O, C

    # --- helper: acceptor N for a CO in residue `res` ---
    def acceptor_N_for_CO(res):
        """
        Return the amide N covalently bonded to the carbonyl C' of `res`,
        approximated as the N of the next residue with the same segid.
        Falls back to N of `res` if no such residue is found.
        """
        N_self = res.atoms.select_atoms("name N")
        segid = res.segid
        idx = res.resindex

        # search forward until segid changes
        for k in range(idx + 1, len(u.residues)):
            r2 = u.residues[k]
            if r2.segid != segid:
                break  # crossed into another chain/segment
            N_next = r2.atoms.select_atoms("name N")
            if len(N_next) > 0:
                return N_next

        # fallback: use N of the CO residue itself (better than empty)
        return N_self

    # --- residue i groups ---
    N_i, H_i, O_i, C_i = backbone_groups(res_i)
    N_acc_i = acceptor_N_for_CO(res_i)

    # --- residue j groups ---
    N_j, H_j, O_j, C_j = backbone_groups(res_j)
    N_acc_j = acceptor_N_for_CO(res_j)

    # --- i as donor, j as acceptor (matches file convention N-resid -> CO-resid) ---
    is_hb_ij, R_ij, theta_ij, phi_ij = hbond_geometry(
        N_i, H_i, O_j, C_j, N_acc_j,
        max_NO=max_NO, min_angle=min_angle
    )
    if is_hb_ij:
        return True, "i->j", R_ij, theta_ij, phi_ij

    # --- j as donor, i as acceptor (symmetric check) ---
    is_hb_ji, R_ji, theta_ji, phi_ji = hbond_geometry(
        N_j, H_j, O_i, C_i, N_acc_i,
        max_NO=max_NO, min_angle=min_angle
    )
    if is_hb_ji:
        return True, "j->i", R_ji, theta_ji, phi_ji

    # no hydrogen bond
    return False, None, None, None, None