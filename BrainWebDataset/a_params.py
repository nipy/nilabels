import os

# Paths

pfo_root = '/Users/sebastiano/a_data/BrainWeb'  # set here the folder where the downloaded dataset is stored.

pfo_raw_in_root   = os.path.join(pfo_root, '0_raw')  # Folder where the 20 x 14 brain web files had been downloaded
pfo_nifti_in_root = os.path.join(pfo_root, 'A_nifti')  # Folder where to create the converted data
pfo_tmp_in_root   = os.path.join(pfo_root, 'z_tmp')  # Temporary folder

pfo_data = os.path.join(pfo_root, 'B_data')

# Params

subjects_num_list = ['04', '05', '06', '18', '20', '38', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                     '51', '52', '53', '54']
names_tissues = ['bck', 'csf', 'gm', 'wm', 'fat', 'muscles', 'muscles_skin', 'skull',
                 'vessels', 'fat2', 'dura', 'marrow']
suffix_tissues = 'v'

name_T1 = 't1w'
suffix_t1 = 'p4'

name_crisp = 'crisp'
suffix_crisp = 'v'

lab_desc = '''################################################
# ITK-SnAP Label Description File for Brain web dataset
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields: 
#    IDX:   Zero-based index 
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description 
################################################
    0     0    0    0        0  0  0   "Background"
    1   255    0    0        1  1  1    "CSF"  
    2     0  255    0        1  1  1    "GM"
    3     0    0  255        1  1  1    "WM"
    4   255  255    0        1  1  1    "Fat"
    5     0  255  255        1  1  1    "Muscle"
    6   102  102  255        1  1  1    "Muscle/Skin"
    7    64  128    0        1  1  1    "Skull"
    8     0    0  205        1  1  1    "Vessels"
    9   205  133   63        1  1  1    "Around fat"
   10   210  180  140        1  1  1    "Dura matter"
   11   102  205  170        1  1  1    "Bone marrow"
   12   200  200  200        1  1  1    "Else"'''
