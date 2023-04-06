import torch
from torch import nn
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore
import evaluation


class Trajectron(nn.Module):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = nn.ModuleDict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[str(node_type)] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[str(node_type)].step_annealers()

    def forward(self, batch, node_type):
        return self.train_loss(batch, node_type)

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[str(node_type)]
        loss = model(inputs=x,
                     inputs_st=x_st_t,
                     first_history_indices=first_history_index,
                     labels=y,
                     labels_st=y_st_t,
                     neighbors=restore(neighbors_data_st),
                     neighbors_edge_value=restore(neighbors_edge_value),
                     robot=robot_traj_st_t,
                     map=map,
                     prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[str(node_type)]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict_and_evaluate_batch(self,
                                   batch, 
                                   node_type,
                                   min_history_timesteps):
        """Predicts from a batch and then evaluates the output, returning the batched errors.
        """

        model = self.node_models_dict[str(node_type)]

        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        ph = y_t.shape[1]

        keep_indices = x_t.shape[1] - first_history_index >= min_history_timesteps

        x = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_t = y_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        keep_indices = keep_indices.to(self.device)

        # Run forward pass
        predictions = model.predict(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=ph,
                                    num_samples=1,
                                    z_mode=True,
                                    gmm_mode=True,
                                    full_dist=False,
                                    output_dists=False)

        # Run forward pass
        y_dists, _  = model.predict(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=ph,
                                    num_samples=1,
                                    z_mode=False,
                                    gmm_mode=False,
                                    full_dist=True,
                                    output_dists=True)

        return evaluation.compute_batch_statistics_pt(predictions, y_t, y_dists=y_dists,
                                                      keep_indices=keep_indices)
        

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                output_dists=False):

        predictions_dict = {}
        dists_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[str(node_type)]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            pred_object = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep,
                                        output_dists=output_dists)

            if output_dists:
                y_dists, predictions = pred_object
            else:
                predictions = pred_object

            predictions_np = predictions.cpu().detach().numpy()
            if output_dists:
                y_dists.set_device(torch.device('cpu'))

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                    if output_dists:
                        dists_dict[ts] = dict()

                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                if output_dists:
                    dists_dict[ts][nodes[i]] = y_dists.get_for_node(i)

        if output_dists:
            return dists_dict, predictions_dict
        else:
            return predictions_dict
