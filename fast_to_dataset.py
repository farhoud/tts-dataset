from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import json
import glob
import sys

def chunk_audio(input_path, segments, output_path='./data/dataset/wavs'):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_path)
    audio = audio.set_channels(1)
    meta = ""

    # Create output directory if it doesn't exist
   
    os.makedirs(output_path, exist_ok=True)

    # Split audio based on timestamps
    i = 0
    for seg in segments:
        start = seg['start'] * 1000
        end = seg['end'] * 1000
        chunk = audio[start:end]

        # Save the chunked audio to a new file
        filename = f'{os.path.splitext(os.path.basename(input_path))[0]}_{i}'
        output_file = os.path.join(output_path, filename)
        chunk.export(output_file+'.wav', format='wav')
        meta = meta + filename + '|' + seg['text'] + '|' + seg['text'] + '\n'

        print(f'Chunk {i+1} saved to {output_file}.wav')
        i = i + 1
    return meta


DIR_PATH=r'/home/farhoud/bahman/*.mp3'

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)>0:
        src_path = args[0] + '*.mp3'
    else:
        src_path = DIR_PATH    
    os.makedirs('./data/dataset/', exist_ok=True)    
    with open('./data/dataset/metadata.txt', 'a', encoding='utf-8') as metadata:
        for item in glob.glob(src_path):
            filename = os.path.splitext(item)[0] + ".json"
            with open(filename, 'r', encoding='utf-8') as file:
                segments = json.load(file)
                meta = chunk_audio(item, segments)
                metadata.write(meta)
