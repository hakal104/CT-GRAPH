def get_organs():
    """
    Returns list of (stacked) structures corresponding to global,
    coarse-level and fine-level nodes.
    """
    
    organs=['global',
     'abdomen',
     'bone',
     'esophagus',
     'heart',
     'mediastinum',
     'trachea',
     'lung',
     'thyroid',
     'spleen',
     'kidney',
     'gallbladder',
     'liver',
     'stomach',
     'pancreas',
     'adrenal_gland',
     'kidney_cyst',
     'portal_vein_and_splenic_vein',
     'vertebrae',
     'humerus',
     'scapula',
     'clavicula',
     'ribs',
     'sternum',
     'costal_cartilages',
     'esophagus',
     'heart',
     'aorta',
     'pulmonary_vein',
     'brachiocephalic_trunk',
     'subclavian_artery',
     'common_carotid_artery',
     'brachiocephalic_vein',
     'superior_vena_cava',
     'inferior_vena_cava',
     'trachea',
     'lung_upper_lobe_left',
     'lung_lower_lobe_left',
     'lung_upper_lobe_right',
     'lung_middle_lobe_right',
     'lung_lower_lobe_right',
     'thyroid_gland']
    
    return organs

def get_h_dict():
    """
    Hierarchical dictionary mapping coarse-level structures to fine-level structures.
    """
    
    hierarchical_region_dict = {
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
           # "esophagus": None
        },
        'heart': {
         #   "heart": None
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
          #  "trachea": None
        },
        'lung': {
            "lung_upper_lobe_left": None,
        "lung_lower_lobe_left": None,
        "lung_upper_lobe_right": None,
        "lung_middle_lobe_right": None,
        "lung_lower_lobe_right": None
    },
    'thyroid': {
       # "thyroid_gland": None
    }
    }
    
    return hierarchical_region_dict