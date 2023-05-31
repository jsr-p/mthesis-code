from dstnx import data_utils

if __name__ == "__main__":
    suffix = "_large"
    nodes = data_utils.load_test(f"nodes{suffix}")
    neighbors = data_utils.load_test(f"neighbors{suffix}")
    loc_neighbors = data_utils.load_test(f"loc_neighbors{suffix}")
