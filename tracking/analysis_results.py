import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'tnl2k' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot'

trackers.extend(trackerlist(name='supersbt', parameter_name='tnl2k_56.6', dataset_name=dataset_name,
                            run_ids=None, display_name='6'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

