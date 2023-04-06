import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.dataset import EnvironmentDataset, collate
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.comm import all_gather
from collections import defaultdict


def train(rank, args):
    if torch.cuda.is_available():
        args.device = f'cuda:{rank}'
        torch.cuda.set_device(rank)
    else:
        args.device = f'cpu'

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius

    # Distributed LR Scaling
    if args.learning_rate is not None:
        hyperparams['learning_rate'] = args.learning_rate
    hyperparams['learning_rate'] *= dist.get_world_size()

    if rank == 0:
        print('-----------------------')
        print('| TRAINING PARAMETERS |')
        print('-----------------------')
        print('| Batch Size: %d' % args.batch_size)
        print('| Eval Batch Size: %d' % args.eval_batch_size)
        print('| Device: %s' % args.device)
        print('| Learning Rate: %s' % hyperparams['learning_rate'])
        print('| Learning Rate Step Every: %s' % args.lr_step)
        print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
        print('| MHL: %s' % hyperparams['minimum_history_length'])
        print('| PH: %s' % hyperparams['prediction_horizon'])
        print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 args.log_tag + time.strftime('-%d_%b_%Y_%H_%M_%S', time.localtime()))

        if rank == 0:
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

            # Save config to model directory
            with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
                json.dump(hyperparams, conf_json)

            log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    # Offline Calculate Training Scene Graphs
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Rank {rank}: Offline calculating scene graphs")
        for i, scene in enumerate(tqdm(train_scenes, desc='Training Scenes', disable=(rank > 0))):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node,
                                       num_workers=args.indexing_workers,
                                       rank=rank)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        train_sampler = data.distributed.DistributedSampler(
            node_type_data_set,
            num_replicas=dist.get_world_size(),
            rank=rank
        )

        node_type_dataloader = data.DataLoader(node_type_data_set,
                                               collate_fn=collate,
                                               pin_memory=False if args.device == 'cpu' else True,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.preprocess_workers,
                                               sampler=train_sampler)
        train_data_loader[node_type_data_set.node_type] = (node_type_dataloader, train_sampler)

    print(f"Rank {rank}: Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        # Offline Calculate Validation Scene Graphs
        if hyperparams['offline_scene_graph'] == 'yes':
            print(f"Rank {rank}: Offline calculating scene graphs")
            for i, scene in enumerate(tqdm(eval_scenes, desc='Validation Scenes', disable=(rank > 0))):
                scene.calculate_scene_graph(eval_env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node,
                                          num_workers=args.indexing_workers,
                                          rank=rank)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            eval_sampler = data.distributed.DistributedSampler(
                node_type_data_set,
                num_replicas=dist.get_world_size(),
                rank=rank
            )

            node_type_dataloader = data.DataLoader(node_type_data_set,
                                                   collate_fn=collate,
                                                   pin_memory=False if args.device == 'cpu' else True,
                                                   batch_size=args.eval_batch_size,
                                                   shuffle=False,
                                                   num_workers=args.preprocess_workers,
                                                   sampler=eval_sampler)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Rank {rank}: Loaded evaluation data from {eval_data_path}")

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)
    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()

    if torch.cuda.is_available():
        trajectron = DDP(trajectron,
                        device_ids=[rank],
                        output_device=rank,
                        find_unused_parameters=True)
        trajectron_module = trajectron.module
    else:
        trajectron_module = trajectron

    print(f'Rank {rank}: Created Training Model.')

    optimizer = dict()
    lr_scheduler = dict()
    step_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])

        if args.lr_step is not None:
            step_scheduler[node_type] = optim.lr_scheduler.StepLR(optimizer[node_type], step_size=args.lr_step, gamma=0.1)

    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(1, args.train_epochs + 1):
        train_dataset.augment = args.augment
        for node_type, (data_loader, data_sampler) in train_data_loader.items():
            data_sampler.set_epoch(epoch)

            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80, unit_scale=dist.get_world_size(), disable=(rank > 0))
            for batch in pbar:
                trajectron_module.set_curr_iter(curr_iter)
                trajectron_module.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                train_loss = trajectron(batch, node_type)
                pbar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss.detach().item():.2f}")
                train_loss.backward()

                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()
                if rank == 0 and not args.debug:
                    log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                          lr_scheduler[node_type].get_last_lr()[0],
                                          curr_iter)
                    log_writer.add_scalar(f"{node_type}/train/loss", train_loss.detach().item(), curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter

            if args.lr_step is not None:
                step_scheduler[node_type].step()
                
        train_dataset.augment = False

        #################################
        #        VISUALIZATION          #
        #################################
        if rank == 0 and (args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0):
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron_module.predict(scene,
                                                        timestep,
                                                        ph,
                                                        min_future_timesteps=ph,
                                                        z_mode=True,
                                                        gmm_mode=True,
                                                        all_z_sep=False,
                                                        full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron_module.predict(scene,
                                                        timestep,
                                                        ph,
                                                        num_samples=20,
                                                        min_future_timesteps=ph,
                                                        z_mode=False,
                                                        full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = trajectron_module.predict(scene,
                                                        timestep,
                                                        ph,
                                                        min_future_timesteps=ph,
                                                        z_mode=True,
                                                        gmm_mode=True,
                                                        all_z_sep=True,
                                                        full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = defaultdict(list)
                    if rank == 0:
                        print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")

                    for batch in tqdm(data_loader, ncols=80, unit_scale=dist.get_world_size(), 
                                      disable=(rank > 0), desc=f'Epoch {epoch} Eval'):
                        eval_loss_node_type = trajectron_module.predict_and_evaluate_batch(batch, node_type, max_hl)
                        for metric, values in eval_loss_node_type.items():
                            eval_loss[metric].append(values.cpu().numpy())

                    if torch.cuda.is_available() and dist.get_world_size() > 1:
                        gathered_values = all_gather(eval_loss)
                        if rank == 0:
                            eval_loss = []
                            for eval_dicts in gathered_values:
                                eval_loss.extend(eval_dicts)

                    if rank == 0:
                        evaluation.log_batch_errors({node_type: eval_loss},
                                                    [node_type],
                                                    ['ade', 'fde', 'nll_mean', 'nll_final'],
                                                    log_writer,
                                                    'eval',
                                                    epoch)

        if rank == 0 and (args.save_every is not None and args.debug is False and epoch % args.save_every == 0):
            model_registrar.save_models(epoch)

        # Waiting for process 0 to be done its evaluation and visualization.
        if torch.cuda.is_available():
            dist.barrier()


def spmd_main(local_rank):
    if torch.cuda.is_available():
        backend = 'nccl'
    else:
        backend = 'gloo'
    
    dist.init_process_group(backend=backend,
                            init_method='env://')

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
        + f"port = {os.environ['MASTER_PORT']} \n", end=''
    )

    train(local_rank, args)


if __name__ == '__main__':
    spmd_main(args.local_rank)
