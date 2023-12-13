import argparse

import jax
import jax.numpy as jnp
import numpy as onp
from haiku._src.data_structures import FlatMap
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task
from dirichlet_sharder import DirichletSharder


def split_data(data, num_clients, num_classes, alpha):
    sharder = DirichletSharder(num_shards=num_clients, num_classes=num_classes, alpha=alpha)
    data_list = []
    for i in range(len(data["image"])):
        data_list.append({"features" : data["image"][i], "labels" : data["label"][i]})
    sharded_data = list(sharder.shard_rows(data_list))

    # I know it can be done in one loop but at this point it doesn't matter as long as it works
    labels_list = []
    for client in sharded_data:
        sample_list = []
        for sample in client[1]:
            sample_list.append(sample["labels"])
        labels_list.append(onp.array(sample_list))

    features_list = []
    for client in sharded_data:
        sample_list = []
        for sample in client[1]:
            sample_list.append(sample["features"])
        features_list.append(onp.array(sample_list))

    return {"image" : features_list, "label" : labels_list}


def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    test_task = get_task(args, is_test=True)

    opt, update = get_optimizer(args)

    data = next(task.datasets.train)

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        splitted_data = split_data(data, args.number_clients, task.datasets.extra_info["num_classes"], args.alpha)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            # Choose clients
            key, key1 = jax.random.split(key)
            chosen_clients_images = jax.random.choice(key1, jnp.array(splitted_data["image"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_labels = jax.random.choice(key1, jnp.array(splitted_data["label"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_data = {"image": chosen_clients_images, "label": chosen_clients_labels}

            # Federated round
            key, key1 = jax.random.split(key)
            opt_state, _ = update(opt_state, key1, FlatMap(chosen_clients_data))

            # Compute and log losses
            params = opt.get_params(opt_state)

            key, key1 = jax.random.split(key)
            loss = task.loss(params, key1, data)

            test_batch = next(test_task.datasets.test)
            try:
                test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
                test_log = {
                    "test loss": test_loss,
                    "test accuracy": test_acc,
                }
            except AttributeError as e:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                if args.needs_state:
                    state = opt.get_state(opt_state)
                    test_loss = test_task.loss(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}


            outer_valid_batch = next(test_task.datasets.outer_valid)
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)

            run.log(to_log)

        run.finish()


def sweep(args):
    def sweep_fn(args=args):
        key = jax.random.PRNGKey(0)

        run = wandb.init(
            project="learned_aggregation_fl", group=args.name, config=vars(args)
        )
        args = argparse.Namespace(**run.config)

        task = get_task(args)
        test_task = get_task(args, is_test=True)

        opt, update = get_optimizer(args)

        data = next(task.datasets.train)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        splitted_data = split_data(data, args.number_clients, task.datasets.extra_info["num_classes"], args.alpha)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            # Choose clients
            key, key1 = jax.random.split(key)
            chosen_clients_images = jax.random.choice(key1, jnp.array(splitted_data["image"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_labels = jax.random.choice(key1, jnp.array(splitted_data["label"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_data = {"image": chosen_clients_images, "label": chosen_clients_labels}

            # Federated round
            key, key1 = jax.random.split(key)
            opt_state, _ = update(opt_state, key1, FlatMap(chosen_clients_data))

            # Compute and log losses
            params = opt.get_params(opt_state)

            key, key1 = jax.random.split(key)
            loss = task.loss(params, key1, data)

            test_batch = next(test_task.datasets.test)
            try:
                test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
                test_log = {
                    "test loss": test_loss,
                    "test accuracy": test_acc,
                }
            except AttributeError as e:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                if args.needs_state:
                    state = opt.get_state(opt_state)
                    test_loss = test_task.loss(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}


            outer_valid_batch = next(test_task.datasets.outer_valid)
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)

            run.log(to_log)

        run.finish()

    sweep_id = wandb.sweep(
        sweep=args.sweep_config, project="learned_aggregation_fl"
    )
    wandb.agent(sweep_id, sweep_fn)
