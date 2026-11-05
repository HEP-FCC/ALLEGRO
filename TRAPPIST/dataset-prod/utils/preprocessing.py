from podio import root_io
import numpy as np
import awkward as ak
import vector
import math
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------------------------------
# Cuts applied for pu0 events:
#   - If any resolved clusters, skip event
#   - If more than 2 reconstructed clusters, skip event
#   - Most energetic (= leading) cluster in UnpairedAugmentedEMBCaloTopoClusters
#   - Exactly 2 MC photons in the event
#   - For each MC photon, z-coordinate of the endpoint of its trajectory within |z| < z_max
#   and radial coordinate r > r_min
#   - Leading photon MC energy < fraction * cluster energy
#
# Cuts applied for gamma events:
#   - If any resolved clusters, skip event
#   - If more than 2 reconstructed clusters, skip event
#   - Most energetic (= leading) cluster in UnpairedAugmentedEMBCaloTopoClusters
#   - Exactly 1 MC photon in the event
#   - z-coordinate of the endpoint of MC photon trajectory within |z| < z_max
#   and radial coordinate r > r_min
#
# ---------------------------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply geometric cuts to EDM4hep files and save selected events in Parquet format."
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the input EDM4hep ROOT file.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Path to the output Parquet file."
    )
    parser.add_argument(
        "--geo-only", action="store_true", help="Apply only geometric cuts."
    )
    parser.add_argument("--bdt-cuts", action="store_true")
    parser.add_argument(
        "--events-per-file",
        type=int,
        help="Number of events to save in each output Parquet file.",
    )
    parser.add_argument(
        "--mode",
        choices=["pi0", "gamma"],
        default=None,
        help="Whether to run on pi0 or gamma files (used for applying cuts)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def ComputeTheta(px, py, pz):
    return math.acos(pz / math.sqrt(px**2 + py**2 + pz**2))


def ComputePhi(px, py):
    return np.sign(float(py)) * math.acos(px / math.sqrt(px**2 + py**2))


def ClusterPosition2Momentum(energy, posx, posy, posz):
    px = energy * posx / math.sqrt(posx**2 + posy**2 + posz**2)
    py = energy * posy / math.sqrt(posx**2 + posy**2 + posz**2)
    pz = energy * posz / math.sqrt(posx**2 + posy**2 + posz**2)
    return px, py, pz


def BDTlikeCuts(
    event,
    mode,
    mc_coll,
    resolved_pi0_coll,
    paired_cluster_coll,
    unpaired_cluster_coll,
    lower_ratioE,
    upper_ratioE,
    deltaR_cut,
    r_min,
    z_max,
):
    mc_particles = set(event.get(mc_coll))
    resolved = set(event.get(resolved_pi0_coll))
    paired = set(event.get(paired_cluster_coll))
    unpaired = set(event.get(unpaired_cluster_coll))
    photon_mcs = [mc for mc in mc_particles if mc.getPDG() == 22]

    for mc in photon_mcs:
        photon_end_x, photon_end_y, photon_end_z = (
            mc.getEndpoint().x,
            mc.getEndpoint().y,
            mc.getEndpoint().z,
        )
        # Require MC photons endpoint within |z| < z_max  and r > r_min
        if (
            abs(photon_end_z) > z_max
            or math.sqrt(photon_end_x**2 + photon_end_y**2) < r_min
        ):
            return None
    if mode == "pi0" and len(photon_mcs) != 2:
        return None  # Exclude Dalitz decays
    for mc in mc_particles:
        if mc.getPDG() == 111 or (
            mc.getPDG() == 22 and mc.getGeneratorStatus() == 1
        ):  # consider only pi0 / prompt photons for truth matching (one per pi0 / gamma event)
            mc_energy = mc.getEnergy()
            mc_theta = ComputeTheta(
                mc.getMomentum().x, mc.getMomentum().y, mc.getMomentum().z
            )
            mc_phi = ComputePhi(mc.getMomentum().x, mc.getMomentum().y)
            logger.debug(
                f"MC particle energy: {mc_energy}, theta: {mc_theta}, phi: {mc_phi}, PDG: {mc.getPDG()}"
            )
    # case: one resolved pi0 found by Pi0RecoTool
    if len(resolved) == 1:
        part = next(iter(resolved))  # get the single element from the set
        part_theta = ComputeTheta(
            part.getMomentum().x, part.getMomentum().y, part.getMomentum().z
        )
        part_phi = ComputePhi(part.getMomentum().x, part.getMomentum().y)
        part_energy = part.getEnergy()
        deltaPhi = abs(mc_phi - part_phi)
        if deltaPhi > math.pi:
            deltaPhi = 2 * math.pi - deltaPhi
        deltaTheta = mc_theta - part_theta
        deltaR = math.sqrt(deltaPhi**2 + deltaTheta**2)
        # if the resolved pi0 passes truth matching is not considered for training
        if (
            deltaR < deltaR_cut
            and part_energy / mc_energy > lower_ratioE
            and part_energy / mc_energy < upper_ratioE
        ):
            return None
        # truth matching failed: take the leading cluster between the ones paired to get a pi0 candidate
        else:
            leading_cl = max(paired, key=lambda cl: cl.getEnergy())
            cl_energy = leading_cl.getEnergy()
            px, py, pz = ClusterPosition2Momentum(
                leading_cl.getEnergy(),
                leading_cl.getPosition().x,
                leading_cl.getPosition().y,
                leading_cl.getPosition().z,
            )
            cl_theta = ComputeTheta(px, py, pz)
            cl_phi = ComputePhi(px, py)
            deltaPhi_cl = abs(mc_phi - cl_phi)
            if deltaPhi_cl > math.pi:
                deltaPhi_cl = 2 * math.pi - deltaPhi_cl
            deltaTheta_cl = mc_theta - cl_theta
            deltaR_cl = math.sqrt(deltaPhi_cl**2 + deltaTheta_cl**2)
            # if the cluster is matched to a truth particle, use it for training: it's a merged pi0 cluster
            if (
                deltaR_cl < deltaR_cut
                and cl_energy / mc_energy > lower_ratioE
                and cl_energy / mc_energy < upper_ratioE
            ):
                selected_cluster = leading_cl
                return selected_cluster
            # failed truth matching with leading cluster: reject for training
            else:
                return None
    # case: no resolved pi0, take leading cluster among unpaired
    elif len(resolved) == 0 and len(unpaired) > 0:
        leading_cl = max(unpaired, key=lambda cl: cl.getEnergy())
        cl_energy = leading_cl.getEnergy()
        px, py, pz = ClusterPosition2Momentum(
            leading_cl.getEnergy(),
            leading_cl.getPosition().x,
            leading_cl.getPosition().y,
            leading_cl.getPosition().z,
        )
        cl_theta = ComputeTheta(px, py, pz)
        cl_phi = ComputePhi(px, py)
        deltaPhi_cl = abs(mc_phi - cl_phi)
        if deltaPhi_cl > math.pi:
            deltaPhi_cl = 2 * math.pi - deltaPhi_cl
        deltaTheta_cl = mc_theta - cl_theta
        deltaR_cl = math.sqrt(deltaPhi_cl**2 + deltaTheta_cl**2)
        # passed truth matching with leading cluster: keep for training
        if (
            deltaR_cl < deltaR_cut
            and cl_energy / mc_energy > lower_ratioE
            and cl_energy / mc_energy < upper_ratioE
        ):
            selected_cluster = leading_cl
            return selected_cluster
        else:
            return None
    else:
        logger.info(
            f"Event has {len(resolved)} resolved pi0s / {len(unpaired)} unpaired clusters."
        )
        return None


def getLeadingUnpairedCluster(
    event, unpaired_cluster_coll, resolved_cluster_coll, cluster_coll
):
    unpaired = set(event.get(unpaired_cluster_coll))
    resolved = set(event.get(resolved_cluster_coll))
    all_clusters = set(event.get(cluster_coll))
    # unpaired =! 1: inefficiency of clustering algorithm / well separated clusters that should be paired in pi0 but are not (inefficiency of Pi0RecoTool)
    # resolved > 0: presence of at least one pi0 candidate reconstructed by Pi0RecoTool
    # all_clusters > 2: more than 2 clusters reconstructed in the event, which is not expected
    if len(unpaired) != 1 or len(resolved) > 0 or len(all_clusters) > 2:
        return None
    cl_lead = max(unpaired, key=lambda cl: cl.getEnergy())
    return cl_lead


def GeoCuts(event, mode, mc_coll, r_min, z_max):
    mc_particles = set(event.get(mc_coll))
    photon_mcs = [mc for mc in mc_particles if mc.getPDG() == 22]
    for mc in photon_mcs:
        photon_end_x, photon_end_y, photon_end_z = (
            mc.getEndpoint().x,
            mc.getEndpoint().y,
            mc.getEndpoint().z,
        )
        # Require endpoint within |z| < z_max  and r > r_min
        if (
            abs(photon_end_z) > z_max
            or math.sqrt(photon_end_x**2 + photon_end_y**2) < r_min
        ):
            return False
    if mode == "pi0":
        if len(photon_mcs) != 2:
            return False  # exclude Dalitz decays
        return True
    elif mode == "gamma":
        if len(photon_mcs) != 1:
            return False
        return True
    else:
        raise ValueError("Mode not set to pi0 or gamma!")


def MCCuts(event, mode, link_coll, mc_coll, sel_cluster, r_min, z_max, fraction):
    # in gamma case only event, r_min, z_max are used, in pi0 case all parameters are used
    mc_particles = set(event.get(mc_coll))
    photon_mcs = [mc for mc in mc_particles if mc.getPDG() == 22]
    for mc in photon_mcs:
        photon_end_x, photon_end_y, photon_end_z = (
            mc.getEndpoint().x,
            mc.getEndpoint().y,
            mc.getEndpoint().z,
        )
        # Require MC photons endpoint within |z| < z_max  and r > r_min
        if (
            abs(photon_end_z) > z_max
            or math.sqrt(photon_end_x**2 + photon_end_y**2) < r_min
        ):
            return False
    if mode == "pi0":
        links = set(event.get(link_coll))
        sel_hits = set(sel_cluster.getHits())
        # Require exactly 2 photons to exclude Dalitz decays
        if len(photon_mcs) != 2:
            return False
        # Require leading particle energy (calculated from hits) < fraction * cluster energy
        photon_dict = {}
        tot_hits_energy = 0.0
        for link in links:
            from_hit = link.getFrom()
            logger.debug(f"from_hit: {from_hit}")
            if from_hit in sel_hits:
                to_mc = link.getTo()
                mc_index = to_mc.getObjectID().index

                logger.debug(f"to_mc: {to_mc}")
                if mc_index not in photon_dict.keys():
                    photon_dict[mc_index] = 0.0
                photon_dict[mc_index] += from_hit.getEnergy() * link.getWeight()
                tot_hits_energy += from_hit.getEnergy() * link.getWeight()
        if (max(photon_dict.values()) / sel_cluster.getEnergy()) > fraction:
            return False
        return True
    elif mode == "gamma":
        if len(photon_mcs) != 1:
            return False
        return True
    else:
        raise ValueError("Mode not set to pi0 or gamma!")


def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = ak.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T


def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


hit_feature_order = [
    "energy",
    "position.x",
    "position.y",
    "position.z",
]


def from_cluster_to_hits_features_and_target(event, mode, cluster, mc_coll):
    hits = cluster.getHits()
    n = len(hits)
    if n == 0:
        raise Exception("Selected cluster has no hits!")
    hit_energy = np.empty(n)
    hit_pos_x = np.empty(n)
    hit_pos_y = np.empty(n)
    hit_pos_z = np.empty(n)
    event_type = np.empty(n)
    mc_part_energy = np.empty(n)
    pi0_opening_angle = np.empty(n)
    pi0_endpoint_distance = np.empty(n)
    mc_particles = set(event.get(mc_coll))
    photon_mcs = [mc for mc in mc_particles if mc.getPDG() == 22]
    if mode == "pi0":
        particles = [mc for mc in mc_particles if abs(mc.getPDG()) == 111]  # pi0 case
        if len(particles) != 1:
            raise Exception(f"Event has {len(particles)} particles, expected 1")
    elif mode == "gamma":
        particles = [mc for mc in mc_particles if mc.getPDG() == 22]  # gamma case
        if len(particles) != 1:
            raise Exception(f"Event has {len(particles)} gammas, expected 1")
    particle = particles[0]
    part_energy = particle.getEnergy()
    if mode == "pi0":
        if len(photon_mcs) != 2:
            raise Exception(f"Event has {len(photon_mcs)} photons, expected exactly 2.")
            return None
        photon1, photon2 = photon_mcs
        # compute distance between photons' endpoints
        end1 = np.array(
            [photon1.getEndpoint().x, photon1.getEndpoint().y, photon1.getEndpoint().z]
        )
        end2 = np.array(
            [photon2.getEndpoint().x, photon2.getEndpoint().y, photon2.getEndpoint().z]
        )
        distance = np.linalg.norm(end1 - end2)
        # compute angle between photons using dot product and convert to degrees
        mom1 = vector.obj(
            px=photon1.getMomentum().x,
            py=photon1.getMomentum().y,
            pz=photon1.getMomentum().z,
        )
        mom2 = vector.obj(
            px=photon2.getMomentum().x,
            py=photon2.getMomentum().y,
            pz=photon2.getMomentum().z,
        )
        dot = mom1.dot(mom2)
        mag_product = mom1.mag * mom2.mag
        angle_rad = math.acos(dot / mag_product)
        angle_deg = math.degrees(angle_rad)
    # Extract data in one loop
    for i, hit in enumerate(hits):
        hit_energy[i] = hit.getEnergy()
        pos = hit.getPosition()
        hit_pos_x[i], hit_pos_y[i], hit_pos_z[i] = pos.x, pos.y, pos.z
        mc_part_energy[i] = part_energy
        if mode == "pi0":
            event_type[i] = 1
            pi0_opening_angle[i] = angle_deg
            pi0_endpoint_distance[i] = distance
        elif mode == "gamma":
            event_type[i] = 0
            pi0_opening_angle[i] = 0
            pi0_endpoint_distance[i] = 0
        else:
            raise Exception("File doesn't have pi0/gamma in the name!")
    hit_dict = {
        "energy": ak.Array(hit_energy),
        "position.x": ak.Array(hit_pos_x),
        "position.y": ak.Array(hit_pos_y),
        "position.z": ak.Array(hit_pos_z),
    }
    return (
        ak.Record(hit_dict),
        event_type,
        mc_part_energy,
        pi0_opening_angle,
        pi0_endpoint_distance,
    )


def build_dummy_array(num, dtype=np.int64):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            ak.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def write_parquet_file(ret, file_index, output_file_name):
    output_name = f"{output_file_name}_{file_index:04d}.parquet"
    n_events = len(ret)
    ret_dict = {k: ak.from_iter([r[k] for r in ret]) for k in ret[0].fields}
    for k in ret_dict.keys():
        if len(ak.flatten(ret_dict[k])) == 0:
            ret_dict[k] = build_dummy_array(len(ret_dict[k]), np.float32)
    ret_record = ak.Record(ret_dict)
    ak.to_parquet(ret_record, output_name, compression="snappy")
    return n_events


# ----------------------------------------------------------------------------------
R_MIN = 2100
Z_MAX = 2350
BDT_LOWER_RATIO_E = 0.6
BDT_UPPER_RATIO_E = 1.5
BDT_DELTA_R_CUT = 0.15
FRACTION = 0.95

if __name__ == "__main__":
    args = parse_arguments()
    if args.debug:
        logging.basicConfig(
            filename="debug.log",
            filemode="w",
            level=logging.DEBUG,
            format="%(levelname)s %(message)s",
        )
        np.set_printoptions(threshold=np.inf)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )
    ev_per_file = args.events_per_file
    output_file = args.output_file
    mode = args.mode
    if mode is None:
        raise ValueError("--mode is required and must be either 'pi0' or 'gamma'")
    if args.input_file is None:
        raise ValueError("--input-file is required")
    else:
        input_file_path = args.input_file

    filename = Path(input_file_path).name
    logger.info(f"Processing file: {filename}")
    podio_reader = root_io.Reader(input_file_path)
    logger.info(f"events in file: {len(podio_reader.get('events'))}")
    ret = []
    file_index = 0
    BDT_cut = 0
    cluster_cut = 0
    mc_cut = 0
    energy_bin = 0
    angle_bin = 0
    distance_bin = 0
    total_events = 0
    for event in podio_reader.get("events"):
        head = event.get("EventHeader")
        ev_number = head[0].getEventNumber()
        if args.bdt_cuts:
            logger.debug(f"Applying BDT-like cuts")
            # ACHTUNG: in old files pairedAugmentedEMBCaloTopoClusters has different name
            selected_cluster = BDTlikeCuts(
                event,
                mode,
                "MCParticles",
                "ResolvedPi0ParticleEMBCaloTopoClusters",
                "PairedAugmentedEMBCaloTopoClusters",
                "UnpairedAugmentedEMBCaloTopoClusters",
                BDT_LOWER_RATIO_E,
                BDT_UPPER_RATIO_E,
                BDT_DELTA_R_CUT,
                R_MIN,
                Z_MAX,
            )
            if selected_cluster is None:
                BDT_cut += 1
                logger.debug(f"Event failed BDT-like cuts: {ev_number}")
                continue
        else:
            logger.debug(f"Selecting leading unpaired cluster")
            selected_cluster = getLeadingUnpairedCluster(
                event,
                "UnpairedAugmentedEMBCaloTopoClusters",
                "ResolvedPi0ParticleEMBCaloTopoClusters",
                "AugmentedEMBCaloTopoClusters",
            )
            if selected_cluster is None:
                cluster_cut += 1
                logger.debug(f"Event has no cluster: {ev_number}")
                continue
            if args.geo_only:
                logger.debug(f"Applying only geometric cuts")
                if not GeoCuts(event, mode, "MCParticles", R_MIN, Z_MAX):
                    mc_cut += 1
                    logger.debug(f"Event failed geo cuts: {ev_number}")
                    continue
            else:
                logger.debug(f"Applying all cuts")
                if not MCCuts(
                    event,
                    mode,
                    "CaloHitMCParticleLinks",
                    "MCParticles",
                    selected_cluster,
                    R_MIN,
                    Z_MAX,
                    FRACTION,
                ):
                    mc_cut += 1
                    logger.debug(f"Event failed MC cuts: {ev_number}")
                    continue
        hit_features, hit_label, mc_energy, pi0_angle, pi0_distance = (
            from_cluster_to_hits_features_and_target(
                event, mode, selected_cluster, "MCParticles"
            )
        )
        # logger.debug(f"hit_features: {hit_features}")
        # logger.debug(f"hit_to_gp: {hit_label}")
        X_hit = get_feature_matrix(hit_features, hit_feature_order)
        ygen_hit = hit_label
        sanitize(X_hit)
        sanitize(ygen_hit)
        sanitize(mc_energy)
        sanitize(pi0_angle)
        sanitize(pi0_distance)
        # logger.debug(f"X_hit: {X_hit}")
        # logger.debug(f"y_: {ygen_hit}")
        this_ev = {
            "X_hit": X_hit,
            "y_event_type": ygen_hit,
            "z_mc_energy": mc_energy,
            "z_pi0_angle": pi0_angle,
            "z_pi0_distance": pi0_distance,
        }
        this_ev = ak.Record(this_ev)
        ret.append(this_ev)
        if len(ret) == ev_per_file:
            total_events += write_parquet_file(ret, file_index, output_file)
            ret.clear()  # reset buffer
            file_index += 1
    # write remaining events
    if len(ret) > 0:
        total_events += write_parquet_file(ret, file_index, output_file)
    logger.info("Finished")
    logger.info(f"BDT cuts not passed: {BDT_cut}")
    logger.info(f"Cluster cut not passed: {cluster_cut}")
    logger.info(f"MC cut not passed: {mc_cut}")
    logger.info(f"Energy cut not passed: {energy_bin}")
    logger.info(f"Angle cut not passed: {angle_bin}")
    logger.info(f"Distance cut not passed: {distance_bin}")
    logger.info(f"Total events passing cuts: {total_events}")
