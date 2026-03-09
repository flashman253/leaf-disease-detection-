import os
from flask import Flask , render_template , jsonify , request
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Bacterial Leaf Spot',
    'Downy Mildew',
    'Healthy Leaf',
    'Mosaic Disease',
    'Powdery Mildew'
]

MODEL_FILES = {
    "densenet": "models/DenseNet_model.pth",
    "efficientnet": "models/EfficientNet_model.pth",
    "mobilenet": "models/MobileNet_model.pth",
    "resnet": "models/resnet_model.pth",
}


def load_model(model_name: str):
    model_path = MODEL_FILES[model_name]

    if model_name == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    elif model_name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def predict_image(image: Image.Image, model):
    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        prob_values, predicted = torch.max(probs, 1)

    predicted_idx = predicted.item()
    predicted_class = class_names[predicted_idx]
    predicted_score = prob_values.item()

    return predicted_class, predicted_score


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        models=list(MODEL_FILES.keys()),
        class_names=class_names,
        prediction=None,
        score=None,
        selected_model=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"message": "No image received"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"message": "No image selected"}), 400

    model_name = request.form.get("model_name")
    if not model_name or model_name not in MODEL_FILES:
        return jsonify({"message": "Invalid or missing model name"}), 400

    image = Image.open(file.stream).convert("RGB")
    model = load_model(model_name)
    predicted_class, score = predict_image(image, model)

    # Return rendered HTML page with prediction
    return render_template(
        "index.html",
        models=list(MODEL_FILES.keys()),
        class_names=class_names,
        prediction=predicted_class,
        score=score,
        selected_model=model_name,
    )


if __name__ == "__main__":
    app.run(debug=True)