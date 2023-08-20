import logging
import os.path
import random
import sys
import copy


class Log(object):
    def __init__(self, args):
        self._logger = None
        self.save = args.logger_path
        self.dataset = args.dataset
        self.time = args.time
        self.__get_logger()

    def __get_logger(self):
        if self._logger is None:
            logger = logging.getLogger("CIP")
            logger.handlers.clear()
            formatter = logging.Formatter('%(message)s')
            if not os.path.exists(f'{self.save}/{self.dataset}-{self.time}'):
                os.mkdir(f'{self.save}/{self.dataset}-{self.time}')
            save_name = f'{self.save}/{self.dataset}-{self.time}/{self.time}-{self.dataset}.txt'
            file_handler = logging.FileHandler(save_name)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            self._logger = logger
            return logger
        else:
            return self._logger

    def info(self, _str):
        self.__get_logger().info(_str)

    def warn(self, _str):
        self.__get_logger().warning(_str)


def decimal_to_bin(x, x_length):
    _x = x
    res = []
    while _x > 0:
        res.append(_x % 2)
        _x = int(_x / 2)
    while len(res) < x_length:
        res.append(0)
    # res.reverse()
    return res


def bin_to_decimal(x):
    res = 0
    for i, value in enumerate(x):
        res += value * (2 ** i)
    return res


def arch_repair(arch, num_cell_nodes):
    empty_layer, i, status = [], 0, True
    while i < num_cell_nodes * 4:
        arch_i_bin = decimal_to_bin(arch[i], int(i / 4) + 2)
        arch_i_bin_len = len(arch_i_bin)
        for j in range(arch_i_bin_len):
            if j in empty_layer:
                arch_i_bin[j] = 0
        arch[i] = bin_to_decimal(arch_i_bin)
        if arch[i] == 0:
            empty_layer.append(int(i / 4) + 2)
            arch[i] = -1
            arch[i + 1] = -1
            arch[i + 2] = -1
            arch[i + 3] = -1
        else:
            status = False
        if arch[i] != -1 and arch[i + 1] == -1:
            pass
        i += 4
    return status, arch


def empty_to_non(arch, i):
    if i == 2:
        return arch
    for j in range(i + 1, 3):
        if arch[j * 4] == -1:
            continue
        bin_topo = decimal_to_bin(arch[j * 4], j + 2)
        if bin_topo[i + 2] != 0:
            pass
        if random.random() < 0.5:
            bin_topo[i + 2] = 1
        arch[j * 4] = bin_to_decimal(bin_topo)
    return arch


def arch_simple(arch):
    arch_len, i = len(arch), 0
    while i < arch_len:
        if arch[i] != -1:
            arch[i] = decimal_to_bin(arch[i], int(i / 4) + 2)
        i += 4
    i = arch_len - 4
    while i >= 0:
        if arch[i] == -1:
            for j in range(i + 4, arch_len, 4):
                arch[j].pop(int(i / 4) + 2)
            arch.pop(i)
            arch.pop(i)
            arch.pop(i)
            arch.pop(i)
            arch_len -= 4
        i -= 4
    return arch


def choose_one_parent(pop_arch, pop_arch_metric):
    count = len(pop_arch)
    index1, index2 = random.randint(0, count - 1), random.randint(0, count - 1)
    while index1 == index2:
        index2 = random.randint(0, count - 1)
    if pop_arch_metric[index1] < pop_arch_metric[index2]:
        return index1, pop_arch[index1]
    elif pop_arch_metric[index2] < pop_arch_metric[index1]:
        return index2, pop_arch[index2]
    else:
        if random.random() < 0.5:
            return index1, pop_arch[index1]
        else:
            return index2, pop_arch[index2]


def delete_inf_metric_arch(archs, archs_metric, archs_pred_metric=None):
    _archs, _archs_metric = copy.deepcopy(archs), copy.deepcopy(archs_metric)
    if archs_pred_metric is not None:
        _archs_pred_metric = copy.deepcopy(archs_pred_metric)
    available_archs, available_metric, available_pred_metric = [], [], []
    _archs_len = len(_archs)
    for i in range(_archs_len):
        if _archs_metric[i] != float('inf'):
            available_archs.append(_archs[i])
            available_metric.append(_archs_metric[i])
            if archs_pred_metric is not None:
                available_pred_metric.append(_archs_pred_metric[i])
    return available_archs, available_metric, available_pred_metric


if __name__ == '__main__':
    print(decimal_to_bin(4, 3))
    print(bin_to_decimal([0, 0, 1]))
    print(arch_simple([-1, -1, -1, -1, 2, 'add', 'appnp', 'elu', 10, 'add', 'generalized_linear', 'tanh']))
    print(decimal_to_bin(-1, 3))
