"""Formula fine-tuning runtime: preserve pretrained visual features on tiny data."""
import sys,runpy
from pathlib import Path
sys.modules['torch']=None
from ppocr.modeling import architectures
original=architectures.build_model
def build_model(config):
 model=original(config)
 for parameter in model.backbone.parameters():parameter.stop_gradient=True
 def keep_backbone_eval(layer,inputs):
  import paddle
  for sublayer in layer.sublayers():
   if isinstance(sublayer,(paddle.nn.BatchNorm1D,paddle.nn.BatchNorm2D,paddle.nn.BatchNorm3D)):sublayer.eval()
 model.backbone.register_forward_pre_hook(keep_backbone_eval)
 return model
architectures.build_model=build_model
script=Path(sys.argv.pop(1));runpy.run_path(str(script),run_name='__main__')
