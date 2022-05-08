import logging
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch.nn.functional as F
import torch.nn as nn
from detectron2.utils.logger import setup_logger
from numpy import deg2rad
import numpy as np
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, SimpleTrainer, create_ddp_model, TrainerBase
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import get_event_storage
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn import dense
import torchvision.transforms as T
import os, weakref, time, torch
from torch.nn.parameter import Parameter
from src.utils.dataloader import MammogramMapper
from src.model.linear import GraphLinear
from detectron2.config import CfgNode as CN
from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["AGRCNN_Trainer", "MassRCNN_Trainer"]

#@META_ARCH_REGISTRY.register()
class AGRCNN(GeneralizedRCNN):
    """
    AGR-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Feature enhancement using graph convolutions
    3. Region proposal generation
    4. Per-region feature extraction and prediction
    """
    def __init__(self, cfg):
        """
        Args:
            node_info: 1. node_count_CC, node_count_MLO, node_features
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(cfg)
        self.class_acc = {"tp":0, "fp":0, "fn": 0, "tn": 0}
        self.seen_so_far = 0
        self.reasoning = True
        self.cc = cfg.MODEL.NODE.CC
        self.mlo = cfg.MODEL.NODE.MLO
        self.encoded_f = cfg.MODEL.NODE.ENCODED_F
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.p2_channel = cfg.MODEL.NODE.F
        self.bipartite_graph_conv1 = GCNConv(cfg.MODEL.NODE.F, cfg.MODEL.NODE.ENCODED_F)
        self.bipartite_graph_conv2 = GCNConv(cfg.MODEL.NODE.ENCODED_F, cfg.MODEL.NODE.F)
        self.bipartite_graph_conv1.lin = GraphLinear(cfg.MODEL.NODE.F, cfg.MODEL.NODE.ENCODED_F, weight_initializer='glorot')
        self.bipartite_graph_conv2.lin = GraphLinear(cfg.MODEL.NODE.ENCODED_F, cfg.MODEL.NODE.F, weight_initializer='glorot')
        self.inception_graph_conv1 = GCNConv(cfg.MODEL.NODE.F, cfg.MODEL.NODE.ENCODED_F)
        self.inception_graph_conv2 = GCNConv(cfg.MODEL.NODE.ENCODED_F, cfg.MODEL.NODE.F)
        self.inception_graph_conv1.lin = GraphLinear(cfg.MODEL.NODE.F, cfg.MODEL.NODE.ENCODED_F, weight_initializer='glorot')
        self.inception_graph_conv2.lin = GraphLinear(cfg.MODEL.NODE.ENCODED_F, cfg.MODEL.NODE.F, weight_initializer='glorot')
        self.inception_dense = nn.Linear(cfg.MODEL.NODE.F, 1, bias=False)
        self.fusion_dense = nn.Linear(2*cfg.MODEL.NODE.F, cfg.MODEL.NODE.F, bias=False)
        self.semantic_dense = nn.Linear(2*cfg.MODEL.NODE.F, 1, bias=False)
        self.class_dense = nn.Linear(200*200, 2)
        depth_conv = nn.Conv2d(in_channels=self.p2_channel, out_channels=self.p2_channel, kernel_size=5, groups=self.p2_channel)
        point_conv = nn.Conv2d(in_channels=self.p2_channel, out_channels=1, kernel_size=1)
        self.depthwise_separable_conv = nn.Sequential(depth_conv, point_conv)
        self.bnorm = nn.BatchNorm1d(200*200)
        self.epsilon = torch.zeros(cfg.MODEL.NODE.CC, cfg.MODEL.NODE.MLO)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4]))
    
    def form_geometric_adjacency(self):
        row_sum = torch.sqrt(torch.sum(self.epsilon, 1))[:, None]
        col_sum = torch.sqrt(torch.sum(self.epsilon, 0))[None, :]
        d = torch.matmul(row_sum, col_sum)
        H_g = torch.div(self.epsilon, d, out=torch.zeros_like(self.epsilon))
        H_g = torch.nan_to_num(H_g, neginf=0.0, posinf=0.0)
        return H_g
    
    def form_semantic_adjacency(self, phi):
        H_s = []
        for i in range(0, len(phi), 3):
          if phi[i].shape[0] == self.cc:
            cc_phi = phi[i]
            mlo_phi = phi[i+1]
          else:
            cc_phi = phi[i+1]
            mlo_phi = phi[i]
          H = torch.zeros(self.cc, self.mlo)
          for j in range(self.cc):
            for k in range(self.mlo):
              x = torch.concat([cc_phi[j], mlo_phi[k]])
              x = self.semantic_dense(x)
              H[j][k] = torch.sigmoid(x)
          H_s.append(H)
        return torch.stack(H_s)
    
    def form_bipartite_graph(self, phi):
        X_B = []
        for i in range(0, len(phi), 3):
          if phi[i].shape[0] == self.cc:
            X_B.append(torch.concat([phi[i], phi[i+1]]))
          else:
            X_B.append(torch.concat([phi[i+1], phi[i]]))
        H_g = self.form_geometric_adjacency()
        H_g = H_g.repeat((self.batch_size, 1, 1))
        H_s = self.form_semantic_adjacency(phi)
        H = H_g*H_s
        bipartite_data = []
        for i in range(self.batch_size):
          H_B = torch.zeros(self.cc+self.mlo, self.cc+self.mlo)
          H_B[:self.cc, self.cc:] = H[i]
          H_B[self.cc:, :self.cc] = H[i].t()
          edge_index, edge_attr = dense_to_sparse(H_B)
          bipartite_data.append(Data(x=X_B[i], edge_index=edge_index, edge_attr=edge_attr))
        return bipartite_data

    def form_inception_graph(self, phi, J_):
        X_I = [torch.concat([phi[i], phi[i+2]]) for i in range(0, len(phi), 3)]
        inception_data = []
        for i in range(self.batch_size):
          sz = X_I[i].shape[0]
          J_hat = torch.zeros(sz, sz)
          J_hat[:sz//2, :sz//2] = torch.eye(sz//2)
          J_hat[sz//2:, sz//2:] = torch.eye(sz//2)
          J_hat[:sz//2, sz//2:] = J_[i]
          J_hat[sz//2:, :sz//2] = J_[i].t()
          edge_index, edge_attr = dense_to_sparse(J_hat)
          inception_data.append(Data(x=X_I[i], edge_index=edge_index, edge_attr=edge_attr))
        return inception_data

    def node_representation_from_feature(self, features_, node_mapping):
        '''
        Form phi first
        '''
        cnt = [torch.max(node_mapping[k][i]) for i in range(self.batch_size) for k in range(3)]
        cnt = [65 if x<=65 else 82 for x in cnt]
        helper_mat = []
        #p2_features = [features_[i]["p2"] for i in range(3)]
        nodes = []
        for i in range(self.batch_size):
          for k in range(3):
            nodes_ = []
            node = torch.clone(node_mapping[k][i][None])
            node = F.interpolate(node[None], scale_factor=(0.5, 0.5), mode="nearest")[0]
            for j in ["p2", "p3", "p4", "p5"]:
              node = F.interpolate(node[None], scale_factor=(0.5, 0.5), mode="nearest")[0]
              nodes_.append(node)
            nodes.append(nodes_)
        for i in range(self.batch_size):
          for k in range(3):
            for node in nodes[i*3+k]:
              cnt_ = cnt[i*3+k]
              adj_ = torch.stack([torch.where(node.flatten()==j, 1, 0) for j in range(1, cnt_+1)], -1)
              adj_ = adj_.type(torch.float32).to(self.device)
              d_ = [torch.count_nonzero(node==j) for j in range(1, cnt_+1)]
              d_ = torch.tensor([1/j if j!=0 else 0 for j in d_])
              d_ = d_.type(torch.float32).to(self.device)
              temp = torch.matmul(adj_, torch.diag(d_))
              helper_mat.append(temp)
        #node_mapping = node_mapping[:, None, :]
        #node_mapping = node_mapping.repeat((1, self.p2_channel, 1, 1))
        #zeros = torch.zeros_like(node_mapping)
        #p2_features = torch.where(node_mapping>0, p2_features, zeros)
        features = [torch.flatten(features_[k][j][i], start_dim=1, end_dim=2) for i in range(self.batch_size) for k in range(3) for j in ["p2", "p3", "p4", "p5"]]
        phi_ = [torch.matmul(helper_mat[i].t(), features[i].t()) for i in range(self.batch_size*12)]
        phi, A, f_E = [], [], []
        for j in range(4):
          phi__, A_, f_E_ = [], [], []
          for i in range(self.batch_size):
            f_E_.append(features[i*12+j])
            A_.append(helper_mat[i*12+j])
            for k in range(3):
              phi__.append(phi_[i*12+k*4+j])
          phi.append(phi__)
          f_E.append(f_E_)
          A.append(A_)
        return phi, A, f_E

    def feature_from_views(self, batched_inputs):
        """
        Args:
              extract auxiliary and contralateral features and train
        """
        images = [x[y].to(self.device) for x in batched_inputs for y in ["image", "auxiliary", "contralateral"]]
        node_mapping = [torch.as_tensor(x[y].astype("float32")).to(self.device) for x in batched_inputs for y in ["map_examined", "map_auxiliary", "map_contralateral"]]
        labels = torch.zeros((self.batch_size,2)).type(torch.float32).to(self.device)
        for i in range(self.batch_size):
          labels[i][batched_inputs[i]["pathology"]] = 1.0
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_e = ImageList.from_tensors([images[i] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        images_a = ImageList.from_tensors([images[i+1] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        images_c = ImageList.from_tensors([images[i+2] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        node_mapping_e = ImageList.from_tensors([node_mapping[i] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        node_mapping_a = ImageList.from_tensors([node_mapping[i+1] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        node_mapping_c = ImageList.from_tensors([node_mapping[i+2] for i in range(0, self.batch_size*3, 3)], self.backbone.size_divisibility)
        flag = bool(self.training)
        with self.eval() and torch.no_grad():
          feature = [self.backbone(images_e.tensor), self.backbone(images_a.tensor), self.backbone(images_c.tensor)]
        if flag:
          self.train()
        J_ = []
        for i in range(self.batch_size):
          self.epsilon += batched_inputs[i]["epsilon"].astype("float32")
          J_.append(torch.as_tensor(batched_inputs[i]["J"].astype("float32")).to(self.device))
        phi, A, f_E = self.node_representation_from_feature(feature, [node_mapping_e.tensor, node_mapping_a.tensor, node_mapping_c.tensor])
        bipartite_data = [self.form_bipartite_graph(phi[i]) for i in range(4)]
        inception_data = [self.form_inception_graph(phi[i], J_) for i in range(4)]
        return bipartite_data, inception_data, A, labels, f_E
    
    def graph_convolutions(self, bipartite_data, inception_data):
        bipartite_node = []
        inception_node = []
        for i in range(self.batch_size):
          data = bipartite_data[i].to(self.device)
          y = self.bipartite_graph_conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
          F.silu(y, True)
          y = self.bipartite_graph_conv2(x=y, edge_index=data.edge_index, edge_weight=data.edge_attr)
          F.silu(y, True)
          bipartite_node.append(y)
          data = inception_data[i].to(self.device)
          y = self.inception_graph_conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
          F.silu(y, True)
          y = self.inception_graph_conv2(x=y, edge_index=data.edge_index, edge_weight=data.edge_attr)
          F.silu(y, True)
          inception_node.append(y)
        return bipartite_node, inception_node
     
    def classify(self, p2_features, label):
        dm = p2_features[0].shape[-2]//2
        ls = self.depthwise_separable_conv(p2_features)
        ls = F.pad(ls, (dm-ls.shape[-1]//2, dm-ls.shape[-1]//2, dm-ls.shape[-2]//2, dm-ls.shape[-2]//2), "constant", 0)
        ls = torch.flatten(ls, 1)
        #ls = self.bnorm(ls)
        w = self.class_dense(ls)
        w = torch.softmax(w, dim=1)
        y = torch.zeros_like(w)
        for i in range(self.batch_size):
          if(w[i][0]>w[i][1]):
            y[i][0] = 1
          else:
            y[i][1] = 1
        loss = self.criterion(w, label)
        return y, {"classification_loss": loss}
    
    def reverse_mapping(self, bipartite_node, inception_node, A):
        psi_bipartite, psi_inception = [], []
        for i in range(self.batch_size):
          if A[i].shape[1] == self.cc:
            psi_bipartite.append(torch.matmul(A[i], bipartite_node[i][:self.cc]))
            psi_inception.append(torch.matmul(A[i], inception_node[i][:self.cc]))
          else:
            psi_bipartite.append(torch.matmul(A[i], bipartite_node[i][self.cc:]))
            psi_inception.append(torch.matmul(A[i], inception_node[i][:self.mlo]))
        return psi_bipartite, psi_inception
    
    def correspondence_reasoning(self, batched_inputs):
      bipartite_data, inception_data, A, labels, f_E = self.feature_from_views(batched_inputs)
      p = []
      shapes = [200, 100, 50, 25]

      for j in range(4):
        bipartite_node, inception_node = self.graph_convolutions(bipartite_data[j], inception_data[j])
        f_B, psi_inception = self.reverse_mapping(bipartite_node, inception_node, A[j])
      
        f_E_hat = []
        for i in range(self.batch_size):
          f_I = torch.sigmoid(self.inception_dense(psi_inception[i]))
          f_I = f_I.t()*f_E[j][i]
          f = torch.reshape(self.fusion_dense(torch.concat([f_I, f_B[i].t()]).t()).t(), (256, shapes[j], -1))
          f_E_hat.append(f[None])
        f_E_hat = torch.concat(f_E_hat)
        p.append(f_E_hat)   
      return p, labels

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        if(self.reasoning):
          p, labels = self.correspondence_reasoning(batched_inputs)
          pkeys = ["p2", "p3", "p4", "p5"]
          for i in range(4):
            features[pkeys[i]] = p[i]
          y, classification_loss = self.classify(features["p2"], labels)
          for i in range(self.batch_size):
            self.seen_so_far+=1
            if y[i][0]==labels[i][0]:
              if labels[i][0]==1:
                self.class_acc["tn"] += 1
              else:
                self.class_acc["tp"] += 1
            else:
              if labels[i][0]==1:
                self.class_acc["fn"] += 1
              else:
                self.class_acc["fp"] += 1
        if(self.seen_so_far%20==0):
          print(self.class_acc, self.seen_so_far)
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if(self.reasoning):
          losses.update(classification_loss)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        bipartite_data, inception_data, node_mapping, labels = self.feature_from_views(batched_inputs) 
        bipartite_node, inception_node = self.graph_convolutions(bipartite_data, inception_data)
        features = self.backbone(images.tensor)
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

class SimplerTrainer(SimpleTrainer):
  def __init__(self, model, data_loader, optimizer):
        super().__init__(model, data_loader, optimizer)

  def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        self.model.train()
        loss_dict = self.model.forward(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()

class AGRCNN_Trainer(DefaultTrainer):
  def __init__(self, train_dataset_name, test_dataset_name, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.INPUT.MIN_SIZE_TRAIN = (800,800)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 300*30
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.META_ARCHITECTURE = "AGRCNN"
    _C = cfg
    _C.MODEL.NODE = CN()
    _C.MODEL.NODE.F = 256
    _C.MODEL.NODE.ENCODED_F = 16
    _C.MODEL.NODE.CC = 65
    _C.MODEL.NODE.MLO = 82
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #self.GRCNN = GCNConv()
    self.mapper = MammogramMapper(cfg)
    TrainerBase.__init__(self)
    self.custom_init(cfg)
    #super().__init__(cfg)

  def custom_init(self, cfg):
    logger = logging.getLogger("detectron2")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        setup_logger()
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
    # Assume these objects must be constructed in this order.
    model = self.build_model(cfg)
    optimizer = self.build_optimizer(cfg, model)
    data_loader = self.build_train_loader(cfg)
    model = create_ddp_model(model, broadcast_buffers=False)
    self._trainer = SimplerTrainer(
        model, data_loader, optimizer
    )
    self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    self.checkpointer = DetectionCheckpointer(
        # Assume you want to save checkpoints together with logs/statistics
        model, 
        cfg.OUTPUT_DIR,
        trainer=weakref.proxy(self),
    )
    self.start_iter = 0
    self.max_iter = cfg.SOLVER.MAX_ITER
    self.cfg = cfg
    self.register_hooks(self.build_hooks())

  def build_train_loader(self, cfg):
    """
    Returns:
        iterable
    It now calls :func:`detectron2.data.build_detection_train_loader`.
    Overwrite it if you'd like a different data loader.
    """
    return build_detection_train_loader(cfg, mapper=self.mapper)
  
  def build_evaluator(self, cfg, dataset_name):
    return 

class MassRCNN_Trainer(DefaultTrainer):
  def __init__(self, train_dataset_name, test_dataset_name, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 600
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    super().__init__(cfg)









