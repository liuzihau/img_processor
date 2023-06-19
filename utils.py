import yaml


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    return opt


def get_data_shape():
    shape_dict = dict()
    shape_dict['features_buffer'] = (1, 99, 128)
    shape_dict['nav_features'] = (1, 64)
    shape_dict['traffic_convention'] = (1, 2)
    shape_dict['desire'] = (1, 100, 8)
    shape_dict['input_imgs'] = (1, 12, 128, 256)
    shape_dict['big_input_imgs'] = (1, 12, 128, 256)
    shape_dict['output'] = (6108,)
    return shape_dict


if __name__ == "__main__":
    get_config('./config/config.yaml')
