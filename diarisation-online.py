from pyannote.audio import Pipeline
from pyannote.audio import Model
import torch

local_model_path = "./model/pyanaudio"

#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
#pipeline = Pipeline.save_pretrained(local_model_path)

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
offline_model = Model.from_pretrained(local_model_path+"/pytorch_model.bin")

for weights, offline_weights in zip(model.parameters(), offline_model.parameters()):
    assert torch.equal(weights, offline_weights)

offline_vad = Pipeline.from_pretrained("./config.yaml")

