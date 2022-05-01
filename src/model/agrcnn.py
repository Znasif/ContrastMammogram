import logging
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch.nn.functional as F
import torch.nn as nn
from detectron2.utils.logger import setup_logger
from numpy import deg2rad
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, SimpleTrainer, create_ddp_model, TrainerBase
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
#from torch_geometric.data import Data, Dataset
#from torch_geometric.utils import dense_to_sparse
#from torch_geometric.nn import GCNConv
import os, weakref, time, torch
from ..utils.dataloader import MammogramMapper
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
        #self.graph_conv1 = GCNConv(cfg.MODEL.NODE.F, 16)
        self.epsilon = torch.zeros((cfg.MODEL.NODE.CC, cfg.MODEL.NODE.MLO))
        self.graph_sigmoid = torch.nn.Sigmoid()
        #self.graph_conv2 = GCNConv(16, cfg.MODEL.NODE.F)
    
    def node_representation_from_feature(self, p2_features, node_mapping):
        for i in node_mapping:
          print(i.shape, end=" ")
        print()

    def feature_from_views(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[y].to(self.device) for y in ["image", "auxiliary", "contralateral"] for x in batched_inputs]
        node_mapping = [torch.as_tensor(x[y].astype("float32")).to(self.device) for y in ["map_examined", "map_auxiliary", "map_contralateral"] for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        with self.eval() and torch.no_grad():
          F = self.backbone(images.tensor)
          self.node_representation_from_feature(F["p2"], node_mapping)
        return images


class SimplerTrainer(SimpleTrainer):
  def __init__(self, model, data_loader, optimizer):
      super().__init__(model, data_loader, optimizer)

  def run_step(self):
    assert self.model.training
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    self.model.feature_from_views(data)
    data_time = time.perf_counter() - start
    self.model.train()
    loss_dict = self.model(data)
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
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 300
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









