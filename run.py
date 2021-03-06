from source.utils import *
from source.swap_test import swap_test_8Q

logger = get_logger(__name__)

if __name__ == '__main__':

    for i in range(8):
        for j in range(i):
            logger.info(j)
            phi_list = [ZERO] * (j - 1)
            phi_list.append(ONE)
            phi_list = phi_list + [ZERO] * (i - j -1)
            phi_list.append(ONE)
            phi_list = phi_list + [ZERO] * (8 - i - 1)

            phi = np.row_stack(phi_list)

            meas = swap_test_8Q(phi)
            logger.info(f'For pair {i, j}, measurements: {meas}')