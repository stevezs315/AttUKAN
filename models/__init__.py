import numpy as np

np.random.seed(0)

from models.deform import *
from models.deform_unet import *
from models.unet import UNet
from models.Attunet import AttU_Net
from models.UKAN import UKAN
from models.DSCNet import DSCNet
from models.RollingUnet import Rolling_Unet_M
from models.mambaunet import MambaUnet
from models.CTFNet import LadderNetv6
from models.IterNet import Iternet
from models.AttUKAN import AttUKAN
from models.BCDUnet import BCDUNet
from models.unet_pp import  UNet_pp
MODELS = {'UNet': UNet,
          'DUNet': DUNetV1V2,
          'AttUNet': AttU_Net,
          'DSCNet': DSCNet,
          'UKAN':UKAN,
          'RollingUnet':Rolling_Unet_M,
          'mambaUNet':MambaUnet,
          'CTFNet':LadderNetv6,
          'IterNet':Iternet,
          'AttUKAN':AttUKAN,
          'BCDUNet':BCDUNet,
          'UNet++':UNet_pp,
          }
