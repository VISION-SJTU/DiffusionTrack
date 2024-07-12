from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.diffusiontrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, pth=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/diffusiontrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if pth is not None:
        params.checkpoint = pth
    else:
        params.checkpoint = '/home/fexie/data/fei/save_pth/diffusiontrack/got/DIFFUSIONTRACK_ep0200.pth.tar'

    print( params.checkpoint)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
