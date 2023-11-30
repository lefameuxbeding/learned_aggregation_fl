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

    splitted_data = {"image": [], "label": []}
    for client in sharded_data:
        image_list = []
        label_list = []
        for sample in client[1]:
            image_list.append(sample["features"])
            label_list.append(sample["labels"])
        splitted_data["image"].append(image_list)
        splitted_data["label"].append(label_list)

    return FlatMap(splitted_data)


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

        splitted_data = split_data(data, args.num_grads, 10, args.alpha) # TODO change num classes dynamically

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            # Choose clients
            key, key1 = jax.random.split(key)
            chosen_clients_images = jax.random.choice(key1, jnp.array(splitted_data["image"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_labels = jax.random.choice(key1, jnp.array(splitted_data["label"]), shape=(int(args.number_clients * args.participation_rate),), replace=False)
            chosen_clients_data = {"image": chosen_clients_images, "label": chosen_clients_labels}

            # Federated round
            key, key1 = jax.random.split(key)
            opt_state, _ = update(opt_state, key1, FlatMap(chosen_clients_data))

            # Compute losses
            params = opt.get_params(opt_state)

            key, key1 = jax.random.split(key)
            loss = task.loss(params, key1, data)

            test_batch = next(test_task.datasets.test)
            key, key1 = jax.random.split(key)
            test_loss = test_task.loss(params, key1, test_batch)

            run.log({"train loss": loss, "test loss": test_loss})

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
