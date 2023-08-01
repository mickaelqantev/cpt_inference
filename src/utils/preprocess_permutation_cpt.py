import torch


def extract_first_4_chars_and_count_unique(input_list):
    first_4_chars_list = [item[:4] for item in input_list]
    unique_set_count = len(set(first_4_chars_list))
    return first_4_chars_list



def extract_first_3_chars_and_count_unique(input_list):
    first_3_chars_list = [item[:3] for item in input_list]
    unique_set_count = len(set(first_3_chars_list))
    return first_3_chars_list




def extract_first_2_chars_and_count_unique(input_list):
    first_2_chars_list = [item[:2] for item in input_list]
    unique_set_count = len(set(first_2_chars_list))
    return first_2_chars_list



def extract_first_1_chars_and_count_unique(input_list):
    first_1_chars_list = [item[:1] for item in input_list]
    unique_set_count = len(set(first_1_chars_list))
    return first_1_chars_list

        
def create_mapping_dict(input_list):
    unique_strings = list(set(input_list))
    mapping_dict = {string: i for i, string in enumerate(unique_strings)}
    return mapping_dict

def map_indices_to_integers(input_list, mapping_dict):
    mapped_indices = [mapping_dict[string] for string in input_list]
    return mapped_indices

def create_permutation_matrix(mapped_indices, num_classes):
    num_mapped_classes = len(mapped_indices)
    permutation_matrix = torch.zeros(num_classes, num_mapped_classes)
    for i, index in enumerate(mapped_indices):
        permutation_matrix[index, i] = 1.0
    return permutation_matrix


def create_permutations(label_transform,device):
    input_list = label_transform.get_classes()
    mapping_dict = create_mapping_dict(input_list)
    mapped_indices = map_indices_to_integers(input_list, mapping_dict)

    # Créer une matrice de permutation pour toutes les étiquettes (1991 classes)
    permutation_matrix_all = create_permutation_matrix(mapped_indices, num_classes=len(mapping_dict)).to(device)

    # Créer une matrice de permutation pour les 4 premiers caractères (nombre de classes réduit)
    input_list_4 = extract_first_4_chars_and_count_unique(label_transform.get_classes())
    mapping_dict_4 = create_mapping_dict(input_list_4)
    mapped_indices_4 = map_indices_to_integers(input_list_4, mapping_dict_4)
    permutation_matrix_4 = create_permutation_matrix(mapped_indices_4, num_classes=len(mapping_dict_4)).to(device)

    # Créer une matrice de permutation pour les 3 premiers caractères (nombre de classes réduit)
    input_list_3 = extract_first_3_chars_and_count_unique(label_transform.get_classes())
    mapping_dict_3 = create_mapping_dict(input_list_3)
    mapped_indices_3 = map_indices_to_integers(input_list_3, mapping_dict_3)
    permutation_matrix_3 = create_permutation_matrix(mapped_indices_3, num_classes=len(mapping_dict_3)).to(device)

    # Créer une matrice de permutation pour les 2 premiers caractères (nombre de classes réduit)
    input_list_2 = extract_first_2_chars_and_count_unique(label_transform.get_classes())
    mapping_dict_2 = create_mapping_dict(input_list_2)
    mapped_indices_2 = map_indices_to_integers(input_list_2, mapping_dict_2)
    permutation_matrix_2 = create_permutation_matrix(mapped_indices_2, num_classes=len(mapping_dict_2)).to(device)

    # Créer une matrice de permutation pour le premier caractère (nombre de classes réduit)
    input_list_1 = extract_first_1_chars_and_count_unique(label_transform.get_classes())
    mapping_dict_1 = create_mapping_dict(input_list_1)
    mapped_indices_1 = map_indices_to_integers(input_list_1, mapping_dict_1)
    permutation_matrix_1 = create_permutation_matrix(mapped_indices_1, num_classes=len(mapping_dict_1)).to(device)
    return permutation_matrix_1,permutation_matrix_2,permutation_matrix_3,permutation_matrix_4,permutation_matrix_all