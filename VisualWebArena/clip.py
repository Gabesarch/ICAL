from PIL import Image
import numpy as np
import ipdb
st = ipdb.set_trace
import torch
# import clip
# from arguments import args
import PIL
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        clip_model = "openai/clip-vit-base-patch32"

        self.model = CLIPModel.from_pretrained(clip_model).to(device).eval()
        self.preprocess = CLIPProcessor.from_pretrained(clip_model)
        print(clip_model)

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def score(self, image=None, texts=None):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

    @torch.no_grad()
    def score_images(self, image_query=None, images=None):

        input_query = self.preprocess(text=None, images=image_query, return_tensors="pt", padding=True).to(device)
        image_features_query = self.model.get_image_features(**input_query)

        if isinstance(images, torch.Tensor):
            image_features = images.to(device)
        else:
            inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
            image_features= self.model.get_image_features(**inputs)

        probs = self.cos_sim(image_features_query, image_features)

        return probs

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
        image_features = self.model.get_image_features(**inputs)

        return image_features

class XVLM():
    def __init__(self):
        from nets.xvlm import XVLMModel

        self.xvlm = XVLMModel(path_checkpoint='checkpoint_9.pth')

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def score(self, images, texts):

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], PIL.Image.Image):
            images = [np.asarray(image) for image in images]

        images = [self.xvlm.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(device)

        texts = [self.xvlm.pre_caption(text, self.xvlm.max_words) for text in texts]
        text_input = self.xvlm.tokenizer(texts, padding='longest', return_tensors="pt").to(device)

        image_embeds, image_atts = self.xvlm.model.get_vision_embeds(images)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.xvlm.model.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.xvlm.model.get_features(image_embeds, text_embeds)
        logits = image_feat @ text_feat.t()

        return logits

    @torch.no_grad()
    def score_images(self, image_query=None, images=None):

        if isinstance(images, torch.Tensor):
            image_features = images.to(device)
        else:
            image_features = self.encode_images(images)
            # if not isinstance(images, list):
            #     images = [images]
            # if isinstance(images[0], PIL.Image.Image):
            #     images = [np.asarray(image) for image in images]
            # images = [self.xvlm.transform(image) for image in images]
            # images = torch.stack(images, dim=0).to(device)
            # image_embeds, image_atts = self.xvlm.model.get_vision_embeds(images)
            # image_features = self.xvlm.model.get_features(image_embeds=image_embeds, text_embeds=None)

        image_features_query = self.encode_images(image_query)
        # if not isinstance(images, list):
        #     images = [image_query]
        # if isinstance(image_query[0], PIL.Image.Image):
        #     images_query = [np.asarray(image) for image in image_query]
        # images_query = [self.xvlm.transform(image) for image in images_query]
        # images_query = torch.stack(images_query, dim=0).to(device)
        # image_embeds_query, image_atts = self.xvlm.model.get_vision_embeds(images_query)
        # image_features_query = self.xvlm.model.get_features(image_embeds=image_embeds_query, text_embeds=None)

        # input_query = self.preprocess(text=None, images=image_query, return_tensors="pt", padding=True).to(device)
        # image_features_query = self.model.get_image_features(**input_query)

        # if isinstance(images, torch.Tensor):
        #     image_features = images.to(device)
        # else:
        #     inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
        #     image_features= self.model.get_image_features(**inputs)

        probs = self.cos_sim(image_features_query, image_features)

        # probs = image_features_query @ image_features.t()

        return probs

    @torch.no_grad()
    def encode_images(self, images):

        if not isinstance(images, list):
            images = [images]
        if isinstance(images[0], PIL.Image.Image):
            images = [np.asarray(image) for image in images]
        images = [self.xvlm.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(device)
        image_embeds, image_atts = self.xvlm.model.get_vision_embeds(images)

        image_features = self.xvlm.model.get_features(image_embeds=image_embeds, text_embeds=None)

        # inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
        # image_features = self.model.get_image_features(**inputs)

        return image_features

class ALIGN:
    def __init__(self):
        from transformers import AlignProcessor, AlignModel

        self.preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()

    @torch.no_grad()
    def score(self, image, texts):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

        