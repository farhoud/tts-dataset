from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch

torch.cuda.empty_cache()

class TranscriberFast():
    def __init__(self):
        self.model = WhisperModel(model_size_or_path="./faster-whisper-small-fa", device="cuda", compute_type="float16")
    def transcribe(self, src, options=None, **kw_args):
        segments, _ = self.model.transcribe(src)
        return list(segments)    

class TranscriberWhisper():
    def __init__(self):
        device = "cpu"
        self.processor = AutoProcessor.from_pretrained("openai/whisper-medium-v3")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            device=device,
        )
    def transcribe(self, src, options=None, **kw_args):
        res = self.pipe(src, return_timestamps=True)
        print(res)




FILE_PATH="./data/tts_1_1_1.mp3"

if __name__ == "__main__":
    tsr = TranscriberFast()
    text = ""
    res = tsr.transcribe(FILE_PATH)
    for item in res:
        text = text + "\n\n" + item.text
    print(text)
    with open('./data/tts_1_1_1.txt', 'w', encoding='utf-8') as f:
        f.write(text)

