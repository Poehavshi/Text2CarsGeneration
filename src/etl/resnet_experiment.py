from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image(r"E:\University\НИР\Project\src\etl\datasets\carconnection\data\Acura_MDX_2019_44_18_290_35_6_77_67_196_20_FWD_7_4_SUV_vRB.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

CAR_IDX = {656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436}
print([weights.meta["categories"][class_id] for class_id in CAR_IDX])