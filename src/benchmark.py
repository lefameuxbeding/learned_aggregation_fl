import argparse

import jax
import jax.numpy as jnp
import numpy as onp
from haiku._src.data_structures import FlatMap
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task


def split_data_into_clients(data, number_clients):
    images = jnp.array(data["image"])
    labels = jnp.array(data["label"])

    def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )
    
    images = split(images, number_clients)
    labels = split(labels, number_clients)

    splitted_data_dict = {"image": [], "label": []}
    for i in range(number_clients):
        splitted_data_dict["image"].append(images[i])
        splitted_data_dict["label"].append(labels[i])

    return FlatMap(splitted_data_dict)


def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    # test_task = get_task(args, is_test=True)

    opt, update = get_optimizer(args)

    data = next(task.datasets.train)

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        splitted_data = split_data_into_clients(data, args.number_clients)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            key, key1 = jax.random.split(key)
            chosen_clients = jax.random.choice(key1, onp.arange(args.number_clients), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_data = {"image": [], "label": []}
            for c in chosen_clients:
                chosen_clients_data["image"].append(splitted_data["image"][c])
                chosen_clients_data["label"].append(splitted_data["label"][c])

            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, FlatMap(chosen_clients_data))

            # key, key1 = jax.random.split(key)
            # params = opt.get_params(opt_state)
            # test_batch = next(test_task.datasets.test)
            # test_loss = test_task.loss(params, key1, test_batch)

            run.log({"train loss": loss}) # , "test loss": test_loss})

        run.finish()


def sweep(args):
    def sweep_fn(args=args):
        key = jax.random.PRNGKey(0)

        run = wandb.init(
            project="learned_aggregation_fl", group=args.name, config=vars(args)
        )
        args = argparse.Namespace(**run.config)

        task = get_task(args)
        # test_task = get_task(args, is_test=True)

        opt, update = get_optimizer(args)

        data = next(task.datasets.train)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        splitted_data = split_data_into_clients(data, args.number_clients)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            key, key1 = jax.random.split(key)
            chosen_clients = jax.random.choice(key1, onp.arange(args.number_clients), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_data = {"image": [], "label": []}
            for c in chosen_clients:
                chosen_clients_data["image"].append(splitted_data["image"][c])
                chosen_clients_data["label"].append(splitted_data["label"][c])

            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, FlatMap(chosen_clients_data))

            # key, key1 = jax.random.split(key)
            # params = opt.get_params(opt_state)
            # test_batch = next(test_task.datasets.test)
            # test_loss = test_task.loss(params, key1, test_batch)

            run.log({"train loss": loss}) # , "test loss": test_loss})

        run.finish()

    sweep_id = wandb.sweep(
        sweep=args.sweep_config, project="learned_aggregation_fl"
    )
    wandb.agent(sweep_id, sweep_fn)
