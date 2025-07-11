from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ArgsDataClass:
    """
    Configuration for neural network models used in MAPLE.
    
    Attributes:
        ct_promblem_type (str): Type of problem (classification/regression)
        encoder_hidden_str (str): Comma-separated string of hidden layer dimensions for encoder
        decoder_hidden_str (str): Comma-separated string of hidden layer dimensions for decoder
        latent_size (int): Dimension of the latent representation
        checkpoint_path (str): Path to model checkpoint file
        feature_size (int): Number of input features (CpG sites)
    """
    ct_promblem_type: str = "classification"
    encoder_hidden_str: str = "1024,1024,256,256,64,64"
    decoder_hidden_str: str = "32,32,16,16"
    latent_size: int = 32
    checkpoint_path: str = "weights/0002_04000t2d.pth.tar"
    feature_size: int = 303212


@dataclass_json
@dataclass
class Config:
    """
    Main configuration class for the MAPLE pipeline.
    
    Attributes:
        args_t2d (ArgsDataClass): Configuration for Type 2 Diabetes prediction model
        args_cvd (ArgsDataClass): Configuration for Cardiovascular Disease prediction model
        args_age (ArgsDataClass): Configuration for age prediction model
        cache_dir (str): Directory for temporary files
    """
    args_t2d: ArgsDataClass = field(default_factory=ArgsDataClass)
    args_cvd: ArgsDataClass = field(default_factory=ArgsDataClass)
    args_age: ArgsDataClass = field(default_factory=ArgsDataClass)
    cache_dir: str = '.cache'
