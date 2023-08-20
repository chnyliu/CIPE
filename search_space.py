gnn_list = [
    "appnp",
    "gcn",
    "gat_1",
    "gat_4",
    "gat_8",
    "sage_mean",
    "sage_max",
    "sage_sum",
    "arma",
    "cheb",
    'gin',
    'gat_sym',
    'cos',
    'linear',
    'generalized_linear'
]

act_list = [
    "tanh", "relu", "elu", "leaky_relu"
]


class SearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {"gnn": gnn_list,
                                 "act": act_list,
                                 "concat_type": ["add", "lstm", "concat", "max"],
                                 'learning_rate': [1e-2, 1e-3, 1e-4, 5e-3, 5e-4],
                                 'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                 'weight_decay_gnns': [1e-2, 1e-3, 1e-4, 1e-5, 5e-3, 5e-4, 5e-5],
                                 'hidden_unit': [32, 64, 128, 256, 512],
                                 'num_cells': [1, 2, 3, 4, 5]
                                 }

    def get_search_space(self, num_cell_nodes):
        actual_actions = []
        for i in range(num_cell_nodes):
            prev_index_list = {
                'name': f"prev_{i}",
                'value': list(range(2 ** (i + 2)))
            }
            actual_actions.append(prev_index_list)

            concat_type = {
                'name': f"cat_{i}",
                'value': self.search_space["concat_type"]
            }
            actual_actions.append(concat_type)

            cur_aggregator = {
                'name': f"gnn_{i}",
                'value': self.search_space["gnn"]
            }
            actual_actions.append(cur_aggregator)

            activate_func = {
                'name': f"activate_{i}",
                'value': self.search_space["act"]
            }
            actual_actions.append(activate_func)

        flag = False
        for key, value in self.search_space.items():
            if key == 'concat_type':
                flag = True
            if flag:
                cur_op = {
                    'name': key,
                    'value': value
                }
                actual_actions.append(cur_op)
        return actual_actions
