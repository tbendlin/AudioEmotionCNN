"""
    Helper file that will record three seconds worth of audio
"""
import pyaudio
import wave

AUDIO_FORM = pyaudio.paInt16
CHANS = 1
SAMPLE_RATE = 44100
CHUNK = 4096
LENGTH = 3
DEV_INDEX = 1
OUTPUT_NAME = "current_sample_test.wav"

class AudioRecorder:
    def __init__(self):
        self.audio = None
        self.wavefile = None
        self.stream = None
    
    def __get_output_name__(self):
        return OUTPUT_NAME
    
    def __start_audio__(self):
        self.audio = pyaudio.PyAudio()
    
    def __stop_audio__(self):
        if self.audio != None:
            self.audio.terminate()
    
    def __open_wav__(self):
        self.wavefile = wave.open(OUTPUT_NAME, 'wb')
        
        if self.audio != None and self.wavefile != None:
            self.wavefile.setnchannels(CHANS)
            self.wavefile.setsampwidth(self.audio.get_sample_size(AUDIO_FORM))
            self.wavefile.setframerate(SAMPLE_RATE)
    
    def __close_wav__(self):
        if self.wavefile != None:
            self.wavefile.close()
    
    def __open_stream__(self):
        if self.audio != None and self.wavefile != None:
            audio_format = self.audio.get_format_from_width(self.wavefile.getsampwidth())
            self.stream = self.audio.open(format=audio_format, rate=SAMPLE_RATE, channels=CHANS, input_device_index=DEV_INDEX, input=True, frames_per_buffer=CHUNK)
    
    def __close_stream__(self):
        if self.stream != None:
            self.stream.stop_stream()
            self.stream.close()
    
    def __get_recording__(self):
        self.__open_wav__()
        self.__open_stream__()
        
        if self.wavefile == None or self.stream == None:
            self.__close_stream__()
            self.__close_wavefile__()
            return False
        
        frames = []

        print("Recording audio...")
        for ii in range(0, int((SAMPLE_RATE/CHUNK) * LENGTH)):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        self.wavefile.writeframes(b''.join(frames))
        
        print("Finished recording!")
        
        self.__close_stream__()
        self.__close_wav__()
    
        return True