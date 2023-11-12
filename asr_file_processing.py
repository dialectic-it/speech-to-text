import warnings
warnings.simplefilter('ignore')

import math
import time
import os
import shutil
import yaml
from pydub import AudioSegment

import torch
import whisper
from whisper.utils import format_timestamp
torch.cuda.empty_cache()

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    
def delete_folder(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError as error: 
        pass

def stt_processing_1_audio(filepath, model, params, transcribe_options):
    name = filepath.split('/')[-1]
    file_format = name.split('.')[-1]
    resultpath1 = os.path.join(params['RESULTPATH'], '{}_net.txt'.format(name))
    resultpath2 = os.path.join(params['RESULTPATH'], '{}_net_ts.txt'.format(name))
    sound = AudioSegment.from_file(filepath)
    
    len_ms = len(sound)
    dt = 1000*params['PART_LEN']
    n_parts = math.ceil(len_ms/dt)
    temp_folder = './temp_folder/'
    create_folder(temp_folder)
    
    for i in range(n_parts):
        print('{}/{}'.format(i+1,n_parts))
        sound_part = sound[dt*i:dt*(i+1)]
        temp_filepath = os.path.join(temp_folder, 'part_{}.{}'.format(i,file_format))
        sound_part.export(temp_filepath, format=file_format)
        dict_pred = model.transcribe(temp_filepath, **transcribe_options)
        
        text = dict_pred["text"]
        if type(text)==list:
            text = text[0]
        with open(resultpath1, "a") as f1:
            f1.writelines(text+"\n======= {}:{}\n".format((i+1)*params['PART_LEN']//60,
                                                         (i+1)*params['PART_LEN']%60))  
        with open(resultpath2, "a") as f2:
            for segment in dict_pred['segments']:
                start_time = format_timestamp(segment['start']+i*params['PART_LEN'],
                                              always_include_hours=True).split('.')[0]
                if (segment['end']+i*params['PART_LEN'])>(i+1)*params['PART_LEN']:
                    end_time = format_timestamp((i+1)*params['PART_LEN'],
                                                always_include_hours=True).split('.')[0]
                else: 
                    end_time = format_timestamp(segment['end']+i*params['PART_LEN'],
                                                always_include_hours=True).split('.')[0]
                text = f"[{start_time}-{end_time}] {segment['text']}"
                f2.writelines(text+'\n')
    delete_folder(temp_folder)
    

if __name__ == "__main__":   
    with open("params.yml", "r") as stream:
        params = yaml.safe_load(stream)

    asr_model = whisper.load_model(name=params['MODEL_NAME'], device=params['DEVICE']) 
    options = dict(language=params['LANG'], beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    
    print('Обрабатываю файл', params['FILEPATH'])
    start_time = time.time()
    stt_processing_1_audio(params['FILEPATH'], asr_model, params, transcribe_options)
    dt = (time.time()-start_time)/60 # min
    print('Обработка завершена, обработка длилась {} минут'.format(round(dt,2)))