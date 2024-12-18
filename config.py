from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ArgsDataClass:
    ct_promblem_type: str = "classification"
    encoder_hidden_str: str =  "1024,1024,256,256,64,64"
    decoder_hidden_str: str = "32,32,16,16"
    latent_size: int = 32
    checkpoint_path: str =  "weights/0002_04000t2d.pth.tar"
    feature_size: int = 303212


@dataclass_json
@dataclass
class Config:
    args_t2d: ArgsDataClass = ArgsDataClass()
    args_cvd: ArgsDataClass = ArgsDataClass()
    args_age: ArgsDataClass = ArgsDataClass()
    device: str = 'cuda'
    cache_dir: str = '.cache'
    input_format: str = ''