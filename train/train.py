# -*- coding: utf-8 -*-
import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.downloaders import download_thorsten_de

output_path = os.path.dirname('./data/training')
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.txt", path="./data/dataset" 
)



audio_config = BaseAudioConfig(
    sample_rate=44100,
    do_trim_silence=True,
    resample=False,
    mel_fmin=0,
    mel_fmax=None 
)
character_config=CharactersConfig(
  characters='ءابتثجحخدذرزسشصضطظعغفقلمنهويِپچژکگیآأؤإئًَُّ',
  punctuations='!(),-.:;? ̠،؛؟‌<>',
  phonemes='ˈˌːˑpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟaegiouwyɪʊ̩æɑɔəɚɛɝɨ̃ʉʌʍ0123456789"#$%*+/=ABCDEFGHIJKLMNOPRSTUVWXYZ[]^_{}',
  pad="<PAD>",
  eos="<EOS>",
  bos="<BOS>",
  blank="<BLNK>",
  characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
  )
config = VitsConfig(
    audio=audio_config,
    run_name="vits_fa_male",
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    save_step=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_language="fa",
    characters=character_config,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    test_sentences=[
    ],
    output_path=output_path,
    datasets=[dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
