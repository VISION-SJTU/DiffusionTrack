from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/home/fexie/data/fei/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/home/fexie/data/fei/data/LaSOT/LaSOTBenchmark'
    settings.network_path = ''    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/fexie/data/fei/data/otb/OTB100'
    settings.prj_dir = '/home/fexie/data/fei/DiffusionTrack/'
    settings.result_plot_path = '/home/fexie/data/fei/DiffusionTrack/save/test_results'
    settings.results_path = '/home/fexie/data/fei/DiffusionTrack/save/test_results'    # Where to store tracking results
    settings.save_dir = '/home/fexie/data/fei/DiffusionTrack/save/test_results'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/fexie/data/fei/data/tnl2k/TNL2K_test_subset'
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

