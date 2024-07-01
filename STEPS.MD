create virtual python environment

pip install 'transformers[torch]'
pip install pyannote.audio
pip install torchvision


delete aud2txt.txt and diarization.rttm from the output folder

Run the Online codes first to download the models
    `python audio2text-online.py`
    `python diarisation-online.py`

check aud2txt.txt and diarization.rttm have been succesfully created


Switch to offline
    Disconnect the computer from the network and run codes offline
    delete aud2txt.txt and diarization.rttm from the output folder

    `python audio2text-offline.py`
    `python diarisation-online.py`

check aud2txt.txt and diarization.rttm have been succesfully created







