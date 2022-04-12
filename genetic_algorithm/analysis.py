import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

result_folder_dict = {#8: '../results/circuit_depth_8_input_8',
                      9: '../results/circuit_8_depth9_run16_10000000',
                      10: '../results/circuit_8_depth10_run17_10000000',
                      11: '../results/circuit_depth_11_input_8',
                      23: '../results/circuit_depth_23_input_16'}

results = np.zeros((len(result_folder_dict), 2))

i = 0

for depth, result_folder in result_folder_dict.items():
    result_files = glob.glob(os.path.join(result_folder, "*.json"))

    succes_list = []

    for file in result_files:
        with open(file, 'r') as json_file:
            res_dict = json.load(json_file)
        succes_list.append(res_dict['succes'])

    success = np.array(succes_list).astype('int32')
    success_rate = success.mean()

    results[i, 0] = depth
    results[i, 1] = success_rate

    i += 1

print(results)

#plot number of overlaps for m=8, depth=9
fig = plt.figure(figsize=(10, 8))

result_folder_dict_9 = {
                      10000: '../results/circuit_8_depth9_run13_10000',
                      100000: '../results/circuit_8_depth9_run14_100000',
                      1000000: '../results/circuit_8_depth9_run15_1000000',
                      10000000: '../results/circuit_8_depth9_run16_10000000'}

result_folder_dict_10 = {10000: '../results/circuit_8_depth10_run20_10000',
                      1000000: '../results/circuit_8_depth10_run18_1000000',
                      10000000: '../results/circuit_8_depth10_run18_10000000'}

result_folder_dict_8 = {10000: '../results/circuit_8_depth8_run21_10000',
                        100000: '../results/circuit_8_depth8_run22_100000',
                        1000000: '../results/circuit_8_depth8_run23_1000000',
                        10000000: '../results/circuit_8_depth8_run24_10000000'
                     }

for result_folder in [ ('Depth = 9', result_folder_dict_9), ('Depth = 10', result_folder_dict_10), ('Depth = 8', result_folder_dict_8),]:
    result_folder_dict = result_folder[1]
    results = np.zeros((len(result_folder_dict), 4))

    i = 0

    for size, folder in result_folder_dict.items():
        result_files = glob.glob(os.path.join(folder, "*.json"))

        cost_list = []

        for file in result_files:
            with open(file, 'r') as json_file:
                res_dict = json.load(json_file)
            if 'cost' in res_dict:
               cost_list.append(28 - int(res_dict['cost']))

        cost = np.array(cost_list)

        results[i, 0] = size
        results[i, 1] = np.mean(cost)
        results[i, 2] = np.quantile(cost, 0.75)
        results[i, 3] = np.quantile(cost, 0.25)

        i += 1

    #plt.plot(results[:, 0], results[:, 1], label=result_folder[0])
    #plt.fill_between(results[:, 0], results[:, 3], results[:, 2], alpha=0.2)
    plt.errorbar(results[:, 0], results[:, 1], yerr=np.stack([results[:, 1]- results[:, 3], results[:, 2] - results[:, 1]], 0),
                 label=result_folder[0])

plt.xscale('log')
plt.legend(fontsize=16, loc='right')
plt.xlabel('Population size', fontsize=14)
plt.ylabel('Average number of overlaps', fontsize=14)

os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/genetic_algo_pop_size')


