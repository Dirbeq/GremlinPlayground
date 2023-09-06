from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("../Fine-Tunning/output_dir")


def predict(img_path):
    # load image from the IAM dataset
    image = Image.open(img_path).convert("RGB")
    # black and white
    image = image.convert("L")
    # reconvert to RGB
    image = image.convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


print("We ", predict("../data/to_predict/we.png"))
print("are ", predict("../data/to_predict/are.png"))
print("working ", predict("../data/to_predict/working.png"))
print("at ", predict("../data/to_predict/at.png"))
print("Euler ", predict("../data/to_predict/Euler.png"))
print("data ", predict("../data/to_predict/data.png"))
print("solution ", predict("../data/to_predict/solution.png"))
print("in ", predict("../data/to_predict/in.png"))
print("Bordeaux ", predict("../data/to_predict/bordeaux.png"))








