import torch.nn.functional as F
import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from search_space import SearchSpace
import copy
import torch_geometric.transforms as T
from network_build import GNN
from model_utils import EarlyStoppingLoss
import gc


def evaluate(output, labels, mask):
    predict = output[mask].max(1)[1].type_as(labels[mask])
    correct = predict.eq(labels[mask]).double()
    acc = correct.sum() / len(labels[mask])
    acc = acc.item()
    return acc


class ModelManager(object):
    def __init__(self, args, logger):
        self.main_args = args
        self.logger = logger
        self.params = {}

        self.epochs = args.epochs
        self.shared_params = None
        self.loss_fn = F.nll_loss
        self.early_stop_manager = None
        self.is_use_early_stop = True if self.main_args.use_early_stop else False
        self.retrain_stage = None
        self.logger.info(f"Dataset: {args.dataset}")
        self.min_metric = 1e+10
        self.min_metric_arch, self.min_metric_arch_stru = None, None
        self.min_metric_arch_test_acc = 0

        if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root='dataset/', name=args.dataset, transform=T.NormalizeFeatures())
            self.data = dataset[0]
        elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
            dataset = WebKB('dataset/', args.dataset)
            self.data = dataset[0]
            self.data.train_mask = self.data.train_mask[:, 0]
            self.data.val_mask = self.data.val_mask[:, 0]
            self.data.test_mask = self.data.test_mask[:, 0]
        elif args.dataset in ['Chameleon']:
            dataset = WikipediaNetwork('dataset/', args.dataset)
            self.data = dataset[0]
            self.data.train_mask = self.data.train_mask[:, 0]
            self.data.val_mask = self.data.val_mask[:, 0]
            self.data.test_mask = self.data.test_mask[:, 0]
        else:
            raise Exception("dataset cannot found")
        self.params["num_node_features"] = self.data.x.shape[1]
        self.params["num_class"] = self.n_classes = self.data.y.max().item() + 1
        self.params["num_edges"] = self.data.edge_index.shape[1]
        self.params["num_nodes"] = self.data.num_nodes
        self.logger.info("Num of edges: {}".format(self.params["num_edges"]))

        space_temp = SearchSpace()
        operations = copy.deepcopy(space_temp.get_search_space(num_cell_nodes=self.main_args.num_cell_nodes))
        self.logger.info(f"Operations: {operations}")
        self.operations = operations

    def train(self, arch=None, actions=None, retrain_stage=None):
        model_actions = actions['action']
        param = actions['hyper_param']

        self.params["lr"] = param[0]
        self.params["in_drop"] = param[1]
        self.params["weight_decay_gnns"] = param[2]
        self.params["num_hidden"] = param[3]
        self.params["num_cells"] = param[4]

        train_epoch = self.epochs
        self.retrain_stage = retrain_stage
        data = self.data
        self.params["in_feats"] = self.in_feats = data.num_features

        model = GNN(model_actions, self.in_feats, self.n_classes, num_cells=self.params["num_cells"],
                    num_hidden=self.params["num_hidden"], dropout=self.params["in_drop"])
        model = model.cuda()
        data = data.cuda()

        stop_epoch = 0
        try:
            early_stop_manager = None
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.params["lr"],
                                         weight_decay=self.params["weight_decay_gnns"])
            if self.is_use_early_stop:
                early_stop_manager = EarlyStoppingLoss(patience=self.main_args.early_stop_size,
                                                       min_epochs=self.main_args.epochs // 2)
            model, metric, test_acc, stop_epoch = self.run_model(self.min_metric, self.logger, self.main_args, model, optimizer, self.loss_fn,
                                                                            data, train_epoch, early_stop_manager)
            if metric < self.min_metric:
                self.min_metric = metric
                self.min_metric_arch_test_acc = test_acc
                self.min_metric_arch = arch
                self.min_metric_arch_stru = actions
        except RuntimeError as e:
            metric = float('inf')
            test_acc = 0.0
            if "cuda" in str(e) or "CUDA" in str(e):
                self.logger.info(f"\t we met cuda OOM; error message: {e}")
            else:
                self.logger.info(f"\t other error: {e}")

        del data
        del model
        del optimizer

        torch.cuda.empty_cache()
        gc.collect()

        return metric, test_acc, stop_epoch

    @staticmethod
    def run_model(min_metric, logger, main_args, model, optimizer, loss_fn, data, epochs, early_stop=None, show_info=False):
        best_performance = 0
        best_metric = float("inf")

        stop_epoch = epochs
        for epoch in range(1, epochs + 1):
            model.train()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            train_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)

            # loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            # metric = loss.item()  # use val loss as metric

            val_acc = evaluate(logits, data.y, data.val_mask)
            metric = 1.0 - val_acc  # use val acc as metric

            test_acc = evaluate(logits, data.y, data.test_mask)

            judge_state = metric < best_metric

            if judge_state:
                best_metric = metric
                best_performance = test_acc
                if best_metric < min_metric:
                    min_metric = best_metric
                    torch.save(model, f'{main_args.logger_path}/{main_args.dataset}-{main_args.time}/model.pth')

            if show_info:
                logger.info(
                    "Epoch {:05d} |Train Loss {:.6f} | Metric {:.6f} | Test_acc {:.6f}".format(
                        epoch, train_loss, metric, test_acc))

            if early_stop is not None:
                early_stop_method = early_stop.on_epoch_end(epoch, metric, train_loss)
                if early_stop_method:
                    stop_epoch = epoch
                    break

        return model, best_metric, best_performance, stop_epoch

    def get_best_arch(self):
        self.logger.info(f"  min metric: {self.min_metric:.6f}, test_acc: {self.min_metric_arch_test_acc:.6f}")
        self.logger.info(f"  arch: {self.min_metric_arch}")
        self.logger.info(f"  structure: {self.min_metric_arch_stru}")
