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



#import torchaudio
#waveform, sample_rate = torchaudio.load("./data/test1.wav")
#audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

offline_vad = Pipeline.from_pretrained("./config2.yaml")
#yaml file updated to point to the pytorch model file (segmentation)


diarization = offline_vad("./data/test1.wav")

#diarization =pipeline("./data/test1.wav")
#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

with open("output/diarization.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# just checking output is the same
#assert (diarization == offline_vad(audio_in_memory))


