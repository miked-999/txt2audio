from pyannote.audio import Pipeline
from pyannote.audio import Model
import torch

local_model_path = "./model/pyanaudio"

with open("./apitoken", 'r') as file:
    key = file.read()
    file.close()

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=key)
offline_model = Model.from_pretrained(local_model_path+"/pytorch_model.bin")

for weights, offline_weights in zip(model.parameters(), offline_model.parameters()):
    assert torch.equal(weights, offline_weights)



#yaml file updated to point to the pytorch model file (segmentation)
offline_vad = Pipeline.from_pretrained("./config2.yaml")



diarization = offline_vad("./data/test1.wav")


# THIS USES THE ONLINE VERSION
#diarization =pipeline("./data/test1.wav")
#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=key)

with open("output/diarization.rttm", "w") as rttm:
    diarization.write_rttm(rttm)




