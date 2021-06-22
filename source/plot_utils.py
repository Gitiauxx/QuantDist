import matplotlib.pyplot as plt
import numpy as np

def plot_overlap_error(results_list, outfolder, n_noise=5):

    fig = plt.figure()
    colors = ['#f39c12', '#8e44ad']

    for i, results in enumerate(results_list):
        name = results[0]
        results = results[1]
        mean = results.mean(1)
        std = results.var(1) ** 0.5

        plt.plot(np.linspace(0, 0.1, num=n_noise), mean, label=f'Mean {name}', color=colors[i])
        plt.fill_between(np.linspace(0, 0.1, num=n_noise), mean - std, mean + std, alpha=0.2, color=colors[i])

    plt.hlines(y=4/3, xmin=0, xmax=0.1, label='Ground truth', color='#34495e')

    plt.xlabel(f'Noise Level', fontsize=14)
    plt.ylabel(f'r12 + r13 + r14 - r23 - r34 - r24', fontsize=14)
    plt.legend(loc='lower left', prop={'size': 14})


    plt.savefig(f'{outfolder}/quasm_simulator_four_qubit.png')

def plot_classic_test(results, outfolder, tag='four', thres=2, gt=None, name='accuracy'):
    fig = plt.figure()
    #plt.subplot(211)

    colors = ['#f39c12', '#8e44ad', '#34495e', '#34495e']

    for i, result in enumerate(results):
        name = result[0]
        plt.scatter(np.arange(result[1].shape[0]), result[1][:, 4], color=colors[i], label=name)

    #plt.axhline(thres, label=f'Classifier boundary', color=colors[1])
    plt.axhline(gt, label=f'Ground truth', color=colors[2])
    plt.legend(bbox_to_anchor=(0, -0.12, 1, 0), ncol=3, loc='lower left', prop={'size': 12})

    plt.xticks([])
    plt.xlabel('')

    if tag == 'four':
        plt.ylabel(f'r12 + r23 + r34 - r14', fontsize=11)
    elif tag == 'six':
        plt.ylabel(f'r12 + r13 + r14 - r23 - r34 - r24', fontsize=11)

    plt.savefig(f'{outfolder}/classifier_{name}_{tag}.png')

def plot_control_curve(results, outfolder, tag='four', type='control', thres=2):
    fig = plt.figure(figsize=(10, 8))

    colors = ['#f39c12', '#8e44ad', '#2471a3', '#34495e', '#27ae60', '#a32471', '#922b21', ]

    def control(x):
        return 2 * np.sin(3 * x / 2) ** 2 + np.sin(5 * x / 2) ** 2 - np.sin(x / 2) ** 2

    for i, result in enumerate(results):
        name = result[0]
        res = result[1]
        res_mean = res[:, 1:].mean(-1)
        error = np.sqrt(res[:, 1:].var(-1))
        plt.errorbar(res[:, 0], res_mean, error, fmt='o', color=colors[i], label=name)

        if i == 0:
            plt.plot(res[:, 0], control(res[:, 0]), color=colors[-1], label='ground truth')

    plt.axhline(thres, label=f'Coherence free threshold', linestyle='--')
    plt.legend( ncol=3, loc='lower right', prop={'size': 12})

    plt.xticks(fontsize=12)
    plt.xlabel(r'$\theta$', fontsize=12)



    plt.yticks(fontsize=12)
    if tag == 'four':
        plt.ylabel(f'Coherence witness', fontsize=20)
    elif tag == 'six':
        plt.ylabel(f'r12 + r13 + r14 - r23 - r34 - r24', fontsize=20)

    plt.savefig(f'{outfolder}/classifier_{type}_{tag}.png')



if __name__ == '__main__':

    results_folder = '../results/four_qubit'
    data = np.load(f'{results_folder}/accuracy_classifier_simulator2.npy')
    data_athens =  np.load(f'{results_folder}/accuracy_classifier_realathens_swap2.npy')
    data_athens2 = np.load(f'{results_folder}/accuracy_classifier_athens_compiled.npy')
    data_simulator2 = np.load(f'{results_folder}/accuracy_classifier_athens_swap2.npy')
    data_layout = np.load(f'{results_folder}/accuracy_classifier_athens_layout.npy')
    data_simulator_belem =  np.load(f'{results_folder}/accuracy_classifier_simulator_belem.npy')
    data_belem_truncated = np.load(f'{results_folder}/accuracy_classifier_real_belem2.npy')
    data_quito_truncated = np.load(f'{results_folder}/accuracy_classifier_real_lima.npy')

    data_swap = np.load(f'{results_folder}/accuracy_classifier_simulator_swap2.npy')
    data_simulator_santiago = np.load(f'{results_folder}/accuracy_classifier_fakesantiago_swap2.npy')


    results = [ ('IBMQ Lima', data_quito_truncated)]
    plot_control_curve(results, results_folder, tag='four', type='control_lima_circuits')

    # data_santiago = np.load(f'{results_folder}/accuracy_classifier2.npy')
    # results = [('athens', data), ('santiago', data_santiago)]
    # plot_classic_test(results, results_folder, gt=1 + np.sqrt(2), tag='four', name='test_bis')

    # data = np.empty((7, 5))
    # data[0, 4] = 0.8310546875
    # data[1, 4] = 0.15625
    # data[2, 4] = 0.171875
    # data[3, 4] = 1.09765625
    # data[4, 4] = 0.6748046875
    # data[5, 4] = 0.1064453125
    # data[6, 4] = 0.8037109375
    #
    # plot_classic_test(data, results_folder, thres=1, gt=4/3, tag='six')


