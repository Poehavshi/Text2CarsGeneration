from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights:
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
resnet50(weights=ResNet50_Weights.DEFAULT)

# Apply it to the input image
orig_img = Image.open(Path('assets') / 'astronaut.jpg')
img_transformed = preprocess(img)