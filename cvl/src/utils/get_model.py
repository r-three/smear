from src.models.resnet_classifier import ResNet_Classifier
from src.models.wae_shapes import WAE_Shapes

def get_model(config, dataset_reader,device):
    '''
    Creates the model based on config
    :param config: configuration for creating model
    :return: model
    '''
    if config.model == "resnet":
        model = ResNet_Classifier(config, dataset_reader, device).to(device)
    elif config.model == 'wae_shapes':
        model = WAE_Shapes(config, device).to(device)
        
    return model