import copy
import numpy as np
import random
import math
from utils import arch_repair, choose_one_parent, empty_to_non, delete_inf_metric_arch
from sklearn.ensemble import RandomForestRegressor


class SurrogateUtils(object):
    def __init__(self, logger, operations, args):
        self.logger = logger
        self.operations = operations
        self.num_cell_nodes = args.num_cell_nodes
        self.pc = args.pc
        self.pm = args.pm
        self.T = args.T
        self.N = args.N

        self.evolve_components = []
        self.probability = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.comp_len = [
            [len(self.operations[0]['value']), len(self.operations[4]['value']), len(self.operations[8]['value'])],
            len(self.operations[1]['value']),
            len(self.operations[2]['value']),
            len(self.operations[3]['value']),
            len(self.operations[13]['value']),
            len(self.operations[14]['value']),
            len(self.operations[15]['value']),
            len(self.operations[16]['value']),
            len(self.operations[17]['value'])
        ]
        probability_record = copy.deepcopy(self.probability)
        probability_record_len = len(probability_record)
        for i in range(probability_record_len):
            probability_record[i] = round(probability_record[i], 4)
        self.probability_records = [probability_record]
        self.evol_comp_records = []

    def initial_arch(self, n):
        def _get_arch():
            status = True
            while status:
                _arch = []
                for op in self.operations:
                    cur_op_num = len(op["value"])
                    cur_op = random.randint(0, cur_op_num - 1)
                    _arch.append(cur_op)
                status, _arch = arch_repair(_arch, self.num_cell_nodes)
                if not status:
                    return _arch

        archs = []
        while len(archs) < n:
            arch = _get_arch()
            if arch not in archs:
                archs.append(arch)
        return archs

    def evolve_arch(self, archs, n, pop_arch, pop_arch_metric):
        self.evolve_components = []
        while len(self.evolve_components) < 1:
            self.evolve_components = []
            prob_len = len(self.probability)
            for i in range(prob_len):
                if random.random() <= self.probability[i]:
                    self.evolve_components.append(i)
        self.evol_comp_records.append(self.evolve_components)
        self.logger.info(f"  selected components: {self.evolve_components}")
        evolve_arch = self.crossover_and_mutation(archs, n, pop_arch, pop_arch_metric)
        return evolve_arch

    def crossover_and_mutation(self, archs, n, pop_arch, pop_arch_metric):
        sub_arch = []
        last_archs = copy.deepcopy(archs)
        while len(sub_arch) < n:
            index1, parent1 = choose_one_parent(pop_arch, pop_arch_metric)
            index2, parent2 = choose_one_parent(pop_arch, pop_arch_metric)
            while index1 == index2:
                index2, parent2 = choose_one_parent(pop_arch, pop_arch_metric)
            child = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            if random.random() < self.pc:
                prob_len = len(self.probability)
                for i in range(prob_len):
                    if i in self.evolve_components:
                        child = self.crossover(child, i, parent1, parent2)
            else:
                pass
            child[0] = self.mutation(child[0], self.evolve_components)
            child[1] = self.mutation(child[1], self.evolve_components)
            for arch in child:
                status, arch = arch_repair(arch, self.num_cell_nodes)
                if not status and arch not in last_archs:
                    sub_arch.append(arch)
                    last_archs.append(arch)
                    if len(sub_arch) == n:
                        break
        return sub_arch

    def crossover(self, child, index, parent0, parent1):
        if index == 0:
            j = 0
            while j <= 8:
                if parent0[j] == parent1[j] == -1:
                    pass
                elif parent0[j] == -1:
                    pass
                elif parent1[j] == -1:
                    pass
                else:
                    child[0][j], child[1][j] = parent1[j], parent0[j]
                j += 4
        elif index == 1 or index == 2 or index == 3:
            j = index
            while j <= index+8:
                if parent0[j] == parent1[j] == -1:
                    pass
                elif parent0[j] == -1:
                    pass
                elif parent1[j] == -1:
                    pass
                else:
                    child[0][j], child[1][j] = parent1[j], parent0[j]
                j += 4
            if index == 1:
                child[0][12], child[1][12] = parent1[12], parent0[12]
        else:
            child[0][index + 9], child[1][index + 9] = parent1[index + 9], parent0[index + 9]
        return child

    def mutation(self, arch, evolve_components):
        i = 0
        while i < 3:
            if arch[i * 4] == -1:
                if random.random() < self.pm:
                    arch[i * 4 + 0] = random.randint(1, self.comp_len[0][i] - 1)
                    arch[i * 4 + 1] = random.randint(0, self.comp_len[1] - 1)
                    arch[i * 4 + 2] = random.randint(0, self.comp_len[2] - 1)
                    arch[i * 4 + 3] = random.randint(0, self.comp_len[3] - 1)
                    arch = empty_to_non(arch, i)
            else:
                for j in range(4):
                    if j in evolve_components and random.random() < self.pm:
                        last_value = arch[i * 4 + j]
                        while arch[i * 4 + j] == last_value:
                            if j == 0:
                                arch[i * 4 + j] = random.randint(0, self.comp_len[0][i] - 1)
                            else:
                                arch[i * 4 + j] = random.randint(0, self.comp_len[j] - 1)
            i += 1
        if 1 in evolve_components and random.random() < self.pm:
            last_value = arch[12]
            while arch[12] == last_value:
                arch[12] = random.randint(0, self.comp_len[1] - 1)
        for j in range(13, len(arch)):
            if (j - 9) in evolve_components and random.random() < self.pm:
                last_value = arch[j]
                while arch[j] == last_value:
                    arch[j] = random.randint(0, self.comp_len[j - 9] - 1)
        return arch

    def predict_and_select_real(self, sub_arch, real_n, archs, archs_metric):
        if float('inf') in archs_metric:
            available_archs, available_metric, _ = delete_inf_metric_arch(archs, archs_metric)
        else:
            available_archs, available_metric = archs, archs_metric
        inputs = np.array(available_archs)
        targets = np.array(available_metric)
        rf_predictor = RandomForestRegressor()
        rf_predictor.fit(inputs, targets)
        rf_prediction = rf_predictor.predict(inputs)
        sub_arch_pred_metric = []
        for arch in sub_arch:
            sub_arch_pred_metric.append(rf_predictor.predict(np.array([arch])).item())
        selected_index = self.select_index(sub_arch_pred_metric, real_n)
        return sub_arch_pred_metric, selected_index, rf_prediction, targets

    def select_index(self, arch_metric, n):
        arch_metric_len = len(arch_metric)
        assert n <= arch_metric_len
        if arch_metric_len == n:
            selected_index = [x for x in range(0, n)]
            return selected_index
        if n <= arch_metric_len // 2:
            selected_length = n * 2
        else:  # >
            selected_length = arch_metric_len
        random_range = []
        while len(random_range) < selected_length:
            num = random.randint(0, arch_metric_len - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index = []
        for i in range(0, selected_length, 2):
            if arch_metric[random_range[i]] <= arch_metric[random_range[i + 1]]:
                selected_index.append(random_range[i])
            else:
                selected_index.append(random_range[i + 1])
        if n > arch_metric_len // 2:
            while len(selected_index) < n:
                index1 = random.randint(0, arch_metric_len - 1)
                while index1 in selected_index:
                    index1 = random.randint(0, arch_metric_len - 1)
                index2 = random.randint(0, arch_metric_len - 1)
                while index2 in selected_index or index2 == index1:
                    index2 = random.randint(0, arch_metric_len - 1)
                if arch_metric[index1] <= arch_metric[index2]:
                    selected_index.append(index1)
                else:
                    selected_index.append(index2)
        return selected_index

    def update_probability(self, last_metric, now_metric, t):
        assert float('inf') not in last_metric
        assert float('inf') not in now_metric
        last_mean, last_min = np.mean(last_metric), np.min(last_metric)
        now_mean, now_min = np.mean(now_metric), np.min(now_metric)
        t_mea = (t + 1) / self.T
        differ = (1 - t_mea) * (now_mean - last_mean) + t_mea * (now_min - last_min)
        if -0.20 < differ < 0:
            differ = -0.20
        elif 0 <= differ < 0.20:
            differ = 0.20
        else:
            pass
        exp = math.exp(t_mea - 0.7)
        rate = 1 - exp * differ
        for i in self.evolve_components:
            self.probability[i] = self.probability[i] * rate
            self.probability[i] = np.min([np.max([self.probability[i], 0]), 1])
        probability_record = copy.deepcopy(self.probability)
        probability_record_len = len(probability_record)
        for i in range(probability_record_len):
            probability_record[i] = round(probability_record[i], 4)
        self.probability_records.append(probability_record)

    def selection(self, pop_arch, pop_arch_metric, sub_arch, sub_arch_metric):
        _arch = pop_arch + sub_arch
        _arch_metric = pop_arch_metric + sub_arch_metric
        new_arch, new_arch_metric = [], []
        random_range = []
        selected_len = int(self.N * 2)
        while len(random_range) < selected_len:
            num = random.randint(0, selected_len - 1)
            if num not in random_range:
                random_range.append(num)
        for i in range(0, selected_len, 2):
            if _arch_metric[random_range[i]] <= _arch_metric[random_range[i+1]]:
                new_arch.append(_arch[random_range[i]])
                new_arch_metric.append(_arch_metric[random_range[i]])
            else:
                new_arch.append(_arch[random_range[i+1]])
                new_arch_metric.append(_arch_metric[random_range[i+1]])
        return new_arch, new_arch_metric

    def transform_to_valid_value(self, arch):
        structure = []
        for i in range(len(arch)):
            if arch[i] == -1:
                structure.append(-1)
            else:
                structure.append(self.operations[i]["value"][arch[i]])
        return structure
