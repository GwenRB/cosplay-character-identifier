# Cosplay character identifier

Drop cosplay photos on a page in your browser and it names the character each
cosplayer is dressed as, with its next guesses. It runs the
[GwenRB/cosplay-characters](https://huggingface.co/GwenRB/cosplay-characters)
model on your own computer: the photos never leave it.

## Run it

You need Python 3.9 or newer.

```bash
pip install -r requirements.txt
python app.py
```

The first start downloads the model (about 180 MB, kept in Hugging Face's
cache for the next time), then opens http://localhost:5000. Drop photos on
the page, or click it to pick some. Each one gets a card: the character, how
sure the model is, the four next guesses, and buttons to search the photo
with Google Lens or look the name up on Google Images.

It works on the processor; with an NVIDIA graphics card and a CUDA build of
PyTorch ([pytorch.org](https://pytorch.org/get-started/locally/)) it is much
faster.

```bash
python app.py --port 8000         # another port
python app.py --no-browser        # don't open the page
```

## What to expect

About 85% of photos of a cosplayer the model never saw get the right
character first, and 91% within its first three guesses. A character trained
on many different cosplayers is recognised far more often than one trained on
a single cosplayer's photos. The model page has the details.

It always answers with one of the characters it knows, even for someone out
of costume or dressed as a character it doesn't know. The page marks answers
under 50% as "not sure".

## From your own code

```python
import timm, torch
from PIL import Image, ImageOps

model = timm.create_model("hf-hub:GwenRB/cosplay-characters", pretrained=True).eval()
transform = timm.data.create_transform(**timm.data.resolve_data_config(model=model))
names = model.pretrained_cfg["label_names"]

with Image.open("photo.jpg") as im:
    x = transform(ImageOps.exif_transpose(im).convert("RGB"))[None]
with torch.no_grad():
    probs = model(x).softmax(1)[0]
for p, i in zip(*probs.topk(3)):
    print(f"{names[i]:<30} {p:.1%}")
```

The transform squashes the whole photo into a square instead of cropping its
centre, as in training. A centre crop also runs, but is less accurate.

## License

GPL-3.0 for this app; the model is Apache-2.0.
