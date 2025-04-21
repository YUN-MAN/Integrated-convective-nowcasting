"""
Utility functions for property extraction.
"""

import networkx as nx
import json
import os
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from random import randint
import matplotlib.pyplot as plt

# Load graph from GML file
def load_graph(filename):
    return nx.read_gml(filename)

# Load JSON data from files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Get unique nodes from merge or split cases
def get_unique_nodes(case_dict, index):
    return sorted(set(path[index] for path in case_dict.values()), key=extract_sort_key)

# Extract sort key for sorting nodes
def extract_sort_key(rain_cell_id):
    date_time, lab, lev, _ = rain_cell_id.split("_")
    return date_time, int(lab), int(lev)

def getCentroidCentroidVector(all_properties, self_cell, other_cell):
    """
    Calculate the centroid-centroid vector between two cells.
    """
    if len(self_cell.split("_")) > 3:
        date_time, lab, lev, ttype = self_cell.split("_")
        cell_key = f"{date_time}_{lab}_{lev}"
    else:
        cell_key = self_cell
    self_centroid = np.array(all_properties[cell_key]['centroid'])

    if len(other_cell.split("_")) > 3:
        date_time, lab, lev, ttype = other_cell.split("_")
        cell_key = f"{date_time}_{lab}_{lev}"
    else:
        cell_key = other_cell
    other_centroid = np.array(all_properties[cell_key]['centroid'])

    return self_centroid - other_centroid

def apportion_cell_properties(all_properties, target_node, subfeatures, diagnostics=False):
    """
    Apportion a rain cell's properties among its subfeatures based on spatial relationships.
    
    This function:
    1. Assigns points from the target cell to each subfeature based on spatial proximity
    2. Calculates apportioned properties (mean2d, area, etc.) for each valid subfeature
    3. Skips subfeatures with insufficient points (< 2) or invalid assignments
    
    Args:
        all_properties (dict): Dictionary containing properties of all rain cells
        target_node (str): ID of the target rain cell to be apportioned
        subfeatures (list): List of subfeature cell IDs to apportion properties to
        diagnostics (bool, optional): Whether to generate diagnostic plots. Defaults to False
    
    Returns:
        tuple: (subfeatures_accounted, subfeatures_properties)
            - subfeatures_accounted (list): List of valid subfeature IDs that were processed
            - subfeatures_properties (dict): Dictionary mapping subfeature IDs to their 
              calculated properties including:
                - mean2d: Mean reflectivity value
                - smjr: Semi-major axis length
                - smnr: Semi-minor axis length
                - area: Cell area
                - altitude: Cell altitude
                - centroid: Cell centroid coordinates
                - pts: Cell point coordinates
                - maxdbz: Maximum reflectivity values
    
    Note:
        Subfeatures are skipped if they:
        - Have no assigned points from the target cell
        - Have fewer than 2 points (insufficient for ellipse calculation)
        - Raise errors during property calculation
    """
    
    ptAssignments = __apportionPts(target_node, subfeatures, all_properties)

    date_time, lab, lev, ttype = target_node.split("_")
    target_cell_key = f"{date_time}_{lab}_{lev}"
    target_node_props = all_properties[target_cell_key]
    
    subfeatsAccounted = []
    subfeats_props = {}
    for iAlloc in range(len(subfeatures)):
        ptIdxs = ptAssignments[iAlloc]
        if not ptIdxs:
            print(f"Warning: no area in self-feature is assigned to subfeature {subfeatures[iAlloc]}")
            continue
        ptIdxs_array = ptIdxs[0]
        subPtSet = (
            np.array(target_node_props['pts'][0])[ptIdxs_array].tolist(),
            np.array(target_node_props['pts'][1])[ptIdxs_array].tolist(),
        )
        # Skip if too few points
        if len(subPtSet[0]) < 2:
            print(f"Warning: too few points in subfeature {subfeatures[iAlloc]}")
            continue
        
        try:
            maxdbz_subset = np.array(target_node_props['maxdbz_values'])[ptIdxs_array]
            mean2d_subset = np.nanmean(maxdbz_subset)
            dbzh_subset = np.array(target_node_props['dbzh_values'])[:, ptIdxs_array]
            altitude_subset, topheight_subset = get_subset_altitude(dbzh_subset)
            centroid, area, semiMajor, semiMinor, _, _ = handle_ellipse_parameters(subPtSet)

            subfeatsAccounted.append(subfeatures[iAlloc])
            subfeats_props[subfeatures[iAlloc]] = {
                'mean2d': mean2d_subset,
                'smjr': semiMajor,
                'smnr': semiMinor,
                'area': area,
                'altitude': altitude_subset,
                'topheight': topheight_subset,
                'centroid': centroid,
                'pts': subPtSet,
                'maxdbz': maxdbz_subset,
            }
        except Exception as e:
            print(f"Warning: Error processing subfeature {subfeatures[iAlloc]}: {str(e)}")
            continue

    if diagnostics:
        target_ptSet = np.array(target_node_props['pts'])
        diagnostics_apportionPts(target_node, 
                                 target_ptSet, 
                                 subfeatsAccounted, 
                                 subfeats_props)
        
    return subfeatsAccounted, subfeats_props

def __apportionPts(target_node, subfeats, all_properties):
    """
    Calculate apportion weights for merge or split cases based on pixel assignments.

    This function distributes properties between cells in merge or split events by:
    1. Identifying the target node and its subfeatures.
    2. Adjusting the target's position to the mean of subfeatures.
    3. Assigning each pixel in the target to the closest subfeature.
    4. Calculating weights based on the proportion of assigned pixels.

    Args:
    all_properties (dict): Dictionary containing properties of all cells.
    path_arr (np.array): Array of cell IDs representing the merge/split path.
    ttype (str): Type of event, either 'merge' or 'split'.

    Returns:
    np.array: Array of weights for each subfeature.
    """

    date_time, lab, lev, ttype = target_node.split("_")
    cell_key = f"{date_time}_{lab}_{lev}"
    target_ptSet = all_properties[cell_key]['pts']
    n_subfeats = len(subfeats)
    subfeats_ptSets = [] # pixel pts of the subfeatures
    for subfeat in subfeats:
        date_time, lab, lev, ttype = subfeat.split("_")
        cell_key = f"{date_time}_{lab}_{lev}"
        subfeat_ptSet = all_properties[cell_key]['pts']
        subfeats_ptSets.append(subfeat_ptSet)

    target_ptSet_disp = np.mean([getCentroidCentroidVector(all_properties, target_node, subfeat) for subfeat in subfeats], axis=0) # mean displacement vector between the target node and the subfeatures
    target_ptSet_m = (np.add(target_ptSet[0], target_ptSet_disp[0]).tolist(), np.add(target_ptSet[1], target_ptSet_disp[1]).tolist()) # move the target node to the mean position of the subfeatures
    target_ptSet_m = np.array(target_ptSet_m)
    target_ptSet = target_ptSet_m

    # Assign each pixel in target_ptSet to the closest subfeat based on minimum Euclidean distance
    # ptSetAssignments is a dictionary with subfeat id as key and the assigned pixels as value
    ptSetAssignments = dict()
    featsToConsider = np.arange(0, n_subfeats, 1)
    for idAlloc in featsToConsider:
        ptSet = subfeats_ptSets[idAlloc]
        # For each pixel in target_ptSet, find the minimal distance to any pixel in the current subfeat (ptSet)
        distMat_min = cdist(target_ptSet.T, np.array(ptSet).T, 'euclidean').min(axis=1)
        # Find the index of the minimal distance for each pixel in target_ptSet
        allocArr = np.argmin(distMat_min, axis=0)
        # Assign the indices to the corresponding subfeat
        ptSetAssignments[idAlloc] = np.array([allocArr])

    # RPP: compute the min distance between each pair of pixels in target_pts and each subfeat_pts
    # find out the closest pixels in target_pts to each object of subfeats.  
    # Find the closest subfeat for each pixel in target_ptSet      
    distMats_min = []
    for ptSet in subfeats_ptSets:
        distMat_min = cdist(target_ptSet.T, np.array(ptSet).T, 'euclidean').min(axis=1)
        distMats_min.append(distMat_min)
    distMats = np.array(distMats_min).T
    allocArr = np.argmin(distMats, axis=1)

    # RPP: calculate the ratio of each allocArr_unique in allocArr
    # calculate the ratio of each allocArr_unique in allocArr
    allocArr_unique = np.unique(allocArr)
    for idAlloc in allocArr_unique:
        pixels_to_this = np.where(allocArr == idAlloc)
        pixels_to_this_rem = np.delete(pixels_to_this, np.where(pixels_to_this == ptSetAssignments[idAlloc]))
        ptSetAssignments[idAlloc] = np.append(ptSetAssignments[idAlloc],
                                                pixels_to_this_rem)
    for idAlloc in featsToConsider:
        ptSetAssignments[idAlloc] = (ptSetAssignments[idAlloc],)
    return ptSetAssignments

def get_subset_altitude(dbzh_values):
    dbzh_values[dbzh_values < 0] = np.nan
    threshold_85 = np.nanpercentile(dbzh_values, 85)
    tmp = np.copy(dbzh_values)
    core_85 = np.where(tmp > threshold_85, tmp, 0)
    centroid_z85, _ = scipy.ndimage.center_of_mass(core_85)
    pts_3d = np.where(tmp > 0)
    topheight = max(pts_3d[0])
    return centroid_z85, topheight

def __getBBox(target_ptSet):
    min0 = np.min(target_ptSet[0])
    max0 = np.max(target_ptSet[0])
    min1 = np.min(target_ptSet[1])
    max1 = np.max(target_ptSet[1])
    return (min0, min1, max0, max1)

def diagnostics_apportionPts(target_node, target_ptSet, subfeats, subfeats_props):
    """
    Create diagnostic visualizations for feature apportioning.
    
    Args:
        target_node: ID of the target feature
        target_ptSet: Point set of the main feature
        subfeats: List of sub-features after splitting
    """
    # Get bounding box for the visualization area
    bbTarget = __getBBox(target_ptSet)
    height = bbTarget[2] - bbTarget[0] + 1
    width = bbTarget[3] - bbTarget[1] + 1
    
    # Create main feature visualization (in blue for better contrast)
    main_img = np.zeros((height, width, 4))
    main_img[
        np.array(target_ptSet[0]) - bbTarget[0],
        np.array(target_ptSet[1]) - bbTarget[1],
        :,
    ] = [0.0, 0.0, 1.0, 1.0]  # Changed to blue for better visibility
    
    # Generate random number for unique filenames
    rn = randint(0, 1000000)
    
    # Save main feature visualization
    plt.imsave(f"{target_node}_main_{rn}.png", main_img)
    
    # Create and save individual sub-feature visualizations
    nSubFeats = len(subfeats)
    for i, subFeat in enumerate(subfeats):
        # Create new image for each sub-feature
        sub_img = np.zeros((height, width, 4))
        
        # Color each sub-feature with increasing intensity of red
        red_intensity = 0.3 + float(i) * 0.7 / float(nSubFeats)  # Range from 0.3 to 1.0
        sub_img[
            np.array(subfeats_props[subFeat]['pts'][0]) - bbTarget[0],
            np.array(subfeats_props[subFeat]['pts'][1]) - bbTarget[1],
            :,
        ] = [red_intensity, 0.0, 0.0, 1.0]  # Full opacity for better visibility
        
        # Save sub-feature visualization
        plt.imsave(
            f"{target_node}_subfeat_{i}_{rn}.png",
            sub_img,
        )
        
        # Optional: Create composite visualization
        main_img[
            np.array(subfeats_props[subFeat]['pts'][0]) - bbTarget[0],
            np.array(subfeats_props[subFeat]['pts'][1]) - bbTarget[1],
            :,
        ] = [red_intensity, 0.0, 0.0, 0.7]  # Add sub-feature overlay
    
    # Save composite visualization
    plt.imsave(f"{target_node}_composite_{rn}.png", main_img)

def handle_ellipse_parameters(pointSet):
    """
    Calculate ellipse parameters from a set of points.
    
    Args:
        pointSet: Tuple of (x_coordinates, y_coordinates)
    """
    # Debug information about input
    if len(pointSet[0]) < 2:
        print(f"Warning: Point set too small - only {len(pointSet[0])} points")
        print(f"Points: {pointSet}")
        raise ValueError("Need at least 2 points to calculate ellipse parameters")

    try:
        # Find mean of xy projected coordinates and get shifted array
        centroid = np.array(pointSet).mean(axis=1)
        
        centredPoints = np.array(pointSet) - centroid[:, np.newaxis]
        
        # Area of storm (and ellipse)
        area = len(pointSet[0])
        
        # Covariance matrix between x and y coordinates
        ptsCov = np.cov(centredPoints)
        
        # Check if covariance matrix is valid
        if np.isnan(ptsCov).any() or np.isinf(ptsCov).any():
            print("Warning: Invalid values in covariance matrix")
            print(f"Centered points:\n{centredPoints}")
            raise ValueError("Invalid covariance matrix")

        d = ptsCov[0, 0]
        e = ptsCov[0, 1]  # off-diagonal are equal
        f = ptsCov[1, 1]
        
        b2m4ac = (d + f) * (d + f) - 4.0 * (d * f - e * e)
        
        if b2m4ac < 0:
            print(f"Warning: Negative discriminant: {b2m4ac}")
            raise ValueError("Cannot compute eigenvalues - negative discriminant")

        lambda1 = 0.5 * ((d + f) + np.sqrt(b2m4ac))  # variance in u-direction
        lambda2 = 0.5 * ((d + f) - np.sqrt(b2m4ac))  # variance in v-direction
        
        sigmaMajor = np.sqrt(lambda1)
        sigmaMinor = np.sqrt(lambda2)
        
        if sigmaMinor < 1.0e-6:
            # Very small minor axis - treating as line
            sigmaMinor = 1.0

        # Calculate semi-major and semi-minor axes
        denominator = np.pi * sigmaMajor * sigmaMinor
        # if denominator <= 0:
        #     print(f"Warning: Invalid denominator: {denominator}")
        #     print(f"sigmaMajor: {sigmaMajor}, sigmaMinor: {sigmaMinor}")
        #     raise ValueError("Cannot calculate semi-axes - invalid denominator")
            
        semiMinor = sigmaMinor * np.sqrt(area / denominator)
        semiMajor = sigmaMajor * np.sqrt(area / denominator)
        
        # Calculate rotation angle
        if np.abs(d + e - lambda1) > 1.0e-6:
            g = (f + e - lambda1) / (d + e - lambda1)
            nu = np.sqrt(1.0 / (1.0 + g * g))
            mu = -g * nu
            tmpAngle = np.arctan2(nu, mu)
            rotAngleSin = np.sin(tmpAngle)
            rotAngleCos = np.cos(tmpAngle)
        else:
            # Circular shape detected - using default rotation
            rotAngleSin = 0.0
            rotAngleCos = 1.0

        return centroid, area, semiMajor, semiMinor, rotAngleSin, rotAngleCos
        
    except Exception as e:
        print(f"\nError in handle_ellipse_parameters:")
        print(f"Point set: {pointSet}")
        print(f"Error message: {str(e)}")
        raise

def organize_properties_for_saving(whole_node_props):
    """
    Separate the properties of whole sequence to individual properties for evolution model input.
    """
    all_key_dict = dict()
    all_path_dict = dict()
    all_mean2d_dict = dict()
    all_smjr_dict = dict()
    all_smnr_dict = dict()
    all_area_dict = dict()
    all_altitude_dict = dict()
    all_topheight_dict = dict()
    for key, values in whole_node_props.items():
        all_key_dict[key] = values["key"]
        all_path_dict[key] = values["path"]
        all_mean2d_dict[key] = values["mean2d"]
        all_smjr_dict[key] = values["smjr"]
        all_smnr_dict[key] = values["smnr"]
        all_area_dict[key] = values["area"]
        all_altitude_dict[key] = values["altitude"]
        all_topheight_dict[key] = values["topheight"]
    return all_key_dict, all_path_dict, all_mean2d_dict, all_smjr_dict, all_smnr_dict, all_area_dict, all_altitude_dict, all_topheight_dict

# Save results to JSON files
def save_to_json(data, folder, filename):
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {filename} to {folder}")