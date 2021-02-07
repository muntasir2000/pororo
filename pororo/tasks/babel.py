from pororo.tasks.automatic_speech_recognition import PororoASR
from pororo.models.wav2vec2.recognizer import BrainWav2Vec2Recognizer
from pororo.models.vad import VoiceActivityDetection


device = 'cpu'

model_path = "/models/checkpoint_best.pt"
dict_path = "/models/dict.ltr.txt"
vad_model_path = download_or_load(
    "misc/vad.pt",
    lang="multi",
)

try:
    import librosa  # noqa
    logging.getLogger("librosa").setLevel(logging.WARN)
except ModuleNotFoundError as error:
    raise error.__class__(
        "Please install librosa with: `pip install librosa`")


vad_model = VoiceActivityDetection(
    model_path=vad_model_path,
    device=device,
)

model = BrainWav2Vec2Recognizer(
    model_path=model_path,
    dict_path=dict_path,
    vad_model=vad_model,
    device=device,
    lang=self.config.lang,
)
asr = PororoASR(model, {})

output = asr.predict("/models/test.wav")
print(output)


