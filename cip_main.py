import argparse
import os
import time
from utils import Log, delete_inf_metric_arch
import torch
import random
import sys
import numpy as np
import torch.backends.cudnn as cudnn
from model_manager import ModelManager
from surrogate_utils import SurrogateUtils
import scipy.stats as stats


def build_args():
    parser = argparse.ArgumentParser(description='Implementation of CIP')
    parser.add_argument('--dataset', type=str, default='Chameleon')
    parser.add_argument('--logger_path', type=str, default='logs')
    parser.add_argument('--time', type=str, default=time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--use_early_stop', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_cell_nodes', type=str, default=3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--early_stop_size', type=int, default=30)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--T', type=int, default=30)
    parser.add_argument('--evaluate_N', type=int, default=30)
    parser.add_argument('--pc', type=float, default=0.9)
    parser.add_argument('--pm', type=float, default=0.2)

    args = parser.parse_args()
    if not os.path.exists(args.logger_path):
        os.mkdir(args.logger_path)
    return args


def init_process(args, logger):
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_feature_name(sur_util):
    feature_names = []
    for i in range(len(sur_util.operations)):
        op = sur_util.operations[i]
        for j in range(len(op["value"])):
            feature_names.append("op {} use {}".format(i, j))
    return feature_names


def train_and_valid(args, logger, sur_util, arch_pool, model_manager):
    valid_pool_metric = []
    test_pool_acc = []
    filtered_arch_pool = []
    arch_id = 0

    for arch in arch_pool:
        gnn_structure = sur_util.transform_to_valid_value(arch)
        actual_action = {
            "action": gnn_structure[: -5],
            "hyper_param": gnn_structure[-5:]
        }
        logger.info(f"  arch_{arch_id + 1:2d}: {actual_action}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        metric, test_score, stop_epoch = model_manager.train(arch, actual_action)
        logger.info(f"\t metric: {metric:.6f}, test_score: {test_score:.6f}, stop_epoch: {stop_epoch}")
        valid_pool_metric.append(metric)
        test_pool_acc.append(test_score)
        filtered_arch_pool.append(arch)

        arch_id += 1

    return model_manager, filtered_arch_pool, valid_pool_metric, test_pool_acc


def main():
    res = []
    for _ in range(5):
        args = build_args()
        args.seed = random.randint(0, 10000)
        logger = Log(args)
        init_process(args, logger)
        logger.info(f"begin_time: {args.time}")
        logger.info(f"args: {args}")
        logger.info(f"args use early stop: {args.use_early_stop}")
        logger.info(f"Pop size:{args.N}; Generation:{args.T}; Seed:{args.seed}")

        model_manager = ModelManager(args, logger)
        operations = model_manager.operations
        sur_util = SurrogateUtils(logger, operations, args)

        archs = []
        archs_metric = []
        archs_test_acc = []

        logger.info("++++++++++ " + time.strftime('%Y%m%d-%H%M%S'))
        logger.info("========== Search the initial generation start ==========")
        logger.info('---- randomly generating architecture pool')
        pop_arch = sur_util.initial_arch(args.N)
        logger.info("---- evaluate the initial population")
        model_manager, pop_arch, pop_arch_metric, pop_arch_test_acc = train_and_valid(args, logger, sur_util, pop_arch, model_manager)
        logger.info(f"---- initial components probability\n  value: {sur_util.probability}")

        archs += pop_arch
        archs_metric += pop_arch_metric
        archs_test_acc += pop_arch_test_acc
        logger.info(f"---- record the best arch")
        model_manager.get_best_arch()

        if float('inf') not in pop_arch_metric:
            last_metric = pop_arch_metric
        else:
            _, last_metric,_ = delete_inf_metric_arch(pop_arch, pop_arch_metric)
        for generation in range(args.T):
            logger.info(f"========== Search generation {generation + 1} start ==========")
            logger.info(f"---- crossover and mutation to generate architecture pool")
            sub_arch = sur_util.evolve_arch(archs, args.N, pop_arch, pop_arch_metric)
            logger.info(f"---- surrogate model predict and select the real evaluate archs")
            sub_arch_pred_metric, selected_index, prediction, targets = sur_util.predict_and_select_real(sub_arch,
                                                                                                           args.evaluate_N,
                                                                                                           archs,
                                                                                                           archs_metric)
            real_eval_arch, real_eval_arch_pred_metric = [], []
            sur_eval_arch, sur_eval_arch_pred_metric = [], []
            for i in range(args.N):
                if i in selected_index:
                    real_eval_arch.append(sub_arch[i])
                    real_eval_arch_pred_metric.append(sub_arch_pred_metric[i])
                else:
                    sur_eval_arch.append(sub_arch[i])
                    sur_eval_arch_pred_metric.append(sub_arch_pred_metric[i])

            logger.info("---- evaluate the selected real-evaluated population")
            model_manager, real_eval_arch, real_eval_arch_metric, real_eval_arch_test_acc = train_and_valid(
                args, logger, sur_util, real_eval_arch, model_manager)

            logger.info("---- record the surrogate model cc")
            if float('inf') not in real_eval_arch_metric:
                avail_real_metric, avail_pred_metric = real_eval_arch_metric, real_eval_arch_pred_metric
            else:
                _, avail_real_metric, avail_pred_metric = delete_inf_metric_arch(real_eval_arch, real_eval_arch_metric, real_eval_arch_pred_metric)
            prediction = np.concatenate((prediction, avail_pred_metric))
            targets = np.concatenate((targets, avail_real_metric))
            rmse = np.sqrt(((prediction - targets) ** 2).mean())
            rho, _ = stats.spearmanr(prediction, targets)
            tau, _ = stats.kendalltau(prediction, targets)
            logger.info(f"  RMSE: {rmse:.6f}, Spearman's Rho: {rho:.6f}, Kendall's Tau: {tau:.6f}")

            archs += real_eval_arch
            archs_metric += real_eval_arch_metric
            archs_test_acc += real_eval_arch_test_acc

            logger.info("---- update the components probability")
            sur_util.update_probability(last_metric, avail_real_metric, generation)
            last_metric = avail_real_metric

            logger.info("---- select the next population")
            sub_arch = sur_eval_arch + real_eval_arch
            sub_arch_metric = sur_eval_arch_pred_metric + real_eval_arch_metric
            pop_arch, pop_arch_metric = sur_util.selection(pop_arch, pop_arch_metric, sub_arch, sub_arch_metric)

            logger.info(f"---- record the best arch")
            model_manager.get_best_arch()

        logger.info(f"==== begin last train")
        model_manager.is_use_early_stop = False
        before_metric = model_manager.min_metric
        _, _, [metric], _ = train_and_valid(args, logger, sur_util, [model_manager.min_metric_arch], model_manager)
        if metric < before_metric:
            logger.info("  Fine last train")
        else:
            logger.info("  Bad last train")
        model_manager.get_best_arch()
        res.append(model_manager.min_metric_arch_test_acc)
        logger.info(f"end_time:{time.strftime('%Y%m%d-%H%M%S')}")
        logger.info(f"==== evaluated gnn len: {len(archs)}")
    print(f'result: {np.mean(res):.6f}+/-{np.std(res):.6f}')
    print(res)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
