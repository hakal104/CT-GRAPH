import numpy as np
from scipy.ndimage import zoom

def interpolate_array(array, target_size):
    """
    Interpolates a 3D NumPy array to a target size.

    Parameters:
        array (np.ndarray): Input 3D array of shape (x, y, z).
        target_size (tuple): Target size (x', y', z').

    Returns:
        np.ndarray: Resized array.
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [t / s for t, s in zip(target_size, array.shape)]
    
    # Perform interpolation
    resized_array = zoom(array, zoom_factors, order=0)  # Cubic interpolation
    
    return resized_array

def map_to_fine_level(orig_arr):
    """
    Maps the labels from the given array (in this case TotalSegmentator labels) 
    to a new set of fine-level labels based on the hierarchical structures dict.

    Parameters:
        orig_arr (np.ndarray): Input label array.

    Returns:
        new_arr (np.ndarray): Input array with adjusted labels.
        fine_structure_list (list): List of fine-level structures.
        fine_level_idx (dict): Dictionary mapping fine-levle structure to label IDs.
    """
    
    total = get_totalseg()
    structure_dict ={valu:key for key, valu in total.items()}
    h_dict = get_h_dict()
    
    fine_structure_list =[]
    fine_level_dict = {}
    fine_level_idx = {}
    
    for coarse_key in h_dict.keys():
        for fine_key in h_dict[coarse_key]:
            fine_structure = h_dict[coarse_key][fine_key]
            fine_structure_list.append(fine_key)
            fine_level_dict[fine_key] = fine_structure
            
    new_arr = np.zeros(orig_arr.shape, dtype=np.uint8)
    for i, label in enumerate(fine_structure_list):
        if fine_level_dict[label] == None:
            ref_label = structure_dict[label] 
            new_arr[orig_arr==ref_label] = i+1
        else:
            for sub_label in fine_level_dict[label]:
                ref_label = structure_dict[sub_label]
                new_arr[orig_arr==ref_label] = i+1
    
    for i,key in enumerate(fine_level_dict.keys()):
        fine_level_idx[key] = i+1
        
    num_fine_nodes = len(fine_structure_list)
        
    return new_arr, fine_level_idx, num_fine_nodes

def map_to_coarse_level(orig_arr, structure_dict):
    """
    Maps the labels from the given array (in this case the fine-level labels) 
    to a new set of coarse-level labels based on the hierarchical structures dict.

    Parameters:
        orig_arr (np.ndarray): Input label array.
        structure_dict (dict): Contains mapping of fine-level structures to their label IDs.

    Returns:
        new_arr (np.ndarray): Input array with adjusted labels.
        coarse_structure_list (list): List of coarse-level structures.
    """
    
    h_dict = get_h_dict()
    coarse_structure_list =[]
    
    for key in h_dict.keys():
        coarse_structure_list.append(key)
    
    num_coarse_nodes = len(coarse_structure_list)
    new_arr = np.zeros(orig_arr.shape, dtype=np.uint8)
    for i, label in enumerate(coarse_structure_list):
        #print(label)

        for sub_label in h_dict[label]:
            ref_label = structure_dict[sub_label]
            new_arr[orig_arr==ref_label] = i+1
    
    return new_arr, num_coarse_nodes

def get_totalseg():
    """
    Dictionary mapping indices to anatomical structures from TotalSegmentator.
    """
    
    total = {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "pancreas",
            8: "adrenal_gland_right",
            9: "adrenal_gland_left",
            10: "lung_upper_lobe_left",
            11: "lung_lower_lobe_left",
            12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right",
            14: "lung_lower_lobe_right",
            15: "esophagus",
            16: "trachea",
            17: "thyroid_gland",
            18: "small_bowel",
            19: "duodenum",
            20: "colon",
            21: "urinary_bladder",
            22: "prostate",
            23: "kidney_cyst_left",
            24: "kidney_cyst_right",
            25: "sacrum",
            26: "vertebrae_S1",
            27: "vertebrae_L5",
            28: "vertebrae_L4",
            29: "vertebrae_L3",
            30: "vertebrae_L2",
            31: "vertebrae_L1",
            32: "vertebrae_T12",
            33: "vertebrae_T11",
            34: "vertebrae_T10",
            35: "vertebrae_T9",
            36: "vertebrae_T8",
            37: "vertebrae_T7",
            38: "vertebrae_T6",
            39: "vertebrae_T5",
            40: "vertebrae_T4",
            41: "vertebrae_T3",
            42: "vertebrae_T2",
            43: "vertebrae_T1",
            44: "vertebrae_C7",
            45: "vertebrae_C6",
            46: "vertebrae_C5",
            47: "vertebrae_C4",
            48: "vertebrae_C3",
            49: "vertebrae_C2",
            50: "vertebrae_C1",
            51: "heart",
            52: "aorta",
            53: "pulmonary_vein",
            54: "brachiocephalic_trunk",
            55: "subclavian_artery_right",
            56: "subclavian_artery_left",
            57: "common_carotid_artery_right",
            58: "common_carotid_artery_left",
            59: "brachiocephalic_vein_left",
            60: "brachiocephalic_vein_right",
            61: "atrial_appendage_left",
            62: "superior_vena_cava",
            63: "inferior_vena_cava",
            64: "portal_vein_and_splenic_vein",
            65: "iliac_artery_left",
            66: "iliac_artery_right",
            67: "iliac_vena_left",
            68: "iliac_vena_right",
            69: "humerus_left",
            70: "humerus_right",
            71: "scapula_left",
            72: "scapula_right",
            73: "clavicula_left",
            74: "clavicula_right",
            75: "femur_left",
            76: "femur_right",
            77: "hip_left",
            78: "hip_right",
            79: "spinal_cord",
            80: "gluteus_maximus_left",
            81: "gluteus_maximus_right",
            82: "gluteus_medius_left",
            83: "gluteus_medius_right",
            84: "gluteus_minimus_left",
            85: "gluteus_minimus_right",
            86: "autochthon_left",
            87: "autochthon_right",
            88: "iliopsoas_left",
            89: "iliopsoas_right",
            90: "brain",
            91: "skull",
            92: "rib_left_1",
            93: "rib_left_2",
            94: "rib_left_3",
            95: "rib_left_4",
            96: "rib_left_5",
            97: "rib_left_6",
            98: "rib_left_7",
            99: "rib_left_8",
            100: "rib_left_9",
            101: "rib_left_10",
            102: "rib_left_11",
            103: "rib_left_12",
            104: "rib_right_1",
            105: "rib_right_2",
            106: "rib_right_3",
            107: "rib_right_4",
            108: "rib_right_5",
            109: "rib_right_6",
            110: "rib_right_7",
            111: "rib_right_8",
            112: "rib_right_9",
            113: "rib_right_10",
            114: "rib_right_11",
            115: "rib_right_12",
            116: "sternum",
            117: "costal_cartilages"
        }
    
    return total

def get_h_dict():
    """
    Hierarchical dictionary mapping coarse-level structures to fine-level structures.
    """
    
    hierarchical_structure_dict = {
        'abdomen': {
            "spleen": None,
            "kidney": ["kidney_left", "kidney_right"],
            "gallbladder": None,
            "liver": None,
            "stomach": None,
            "pancreas": None,
            "adrenal_gland": ["adrenal_gland_left", "adrenal_gland_right"],
            "kidney_cyst": ["kidney_cyst_left", "kidney_cyst_right"],
            "portal_vein_and_splenic_vein": None
        },
        'bone': {
            "vertebrae": [
                "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", 
                "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", 
                "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", 
                "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", 
                "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"
            ],
            "humerus": ["humerus_left", "humerus_right"],
            "scapula": ["scapula_left", "scapula_right"],
            "clavicula": ["clavicula_left", "clavicula_right"],
            "ribs": [
                "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", 
                "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8", 
                "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
                "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", 
                "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8", 
                "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12"
            ],
            "sternum": None,
            "costal_cartilages": None
        },
        'esophagus': {
            "esophagus": None
        },
        'heart': {
            "heart": None
        },
        'mediastinum': {
            "aorta": None,
            "pulmonary_vein": None,
            "brachiocephalic_trunk": None,
            "subclavian_artery": ["subclavian_artery_right", "subclavian_artery_left"],
            "common_carotid_artery": ["common_carotid_artery_right", "common_carotid_artery_left"],
            "brachiocephalic_vein": ["brachiocephalic_vein_left", "brachiocephalic_vein_right"],
            "superior_vena_cava": None,
            "inferior_vena_cava": None
        },
        'trachea': {
            "trachea": None
        },
        'lung': {
            "lung_upper_lobe_left": None,
        "lung_lower_lobe_left": None,
        "lung_upper_lobe_right": None,
        "lung_middle_lobe_right": None,
        "lung_lower_lobe_right": None
    },
    'thyroid': {
        "thyroid_gland": None
    }
    }


    return hierarchical_structure_dict

def count_level_nodes():
    """
    Returns dictionary containing number of global, coarse-level 
    and fine-level nodes.
    """
    
    h_dict = get_h_dict()
    num_nodes = {}  
    num_nodes['global'] = 1
    num_nodes['coarse'] = len([key for key in h_dict.keys()])
    num_nodes['fine'] = np.sum([len(key) for key in h_dict.values()])
    num_nodes['all'] = num_nodes['global']+num_nodes['coarse']+num_nodes['fine']-4
    
    return num_nodes
    