"""
    Main script file that controls all functionality
"""
import time
import vlc
import datetime
from ModelRunner import ModelRunner
from MotorDriver import MotorDriver
from RecordAudio import AudioRecorder
from PreprocessAudio import preprocess_and_create_spectrogram

LULLABY_FILE = "../Twinkle.mp3"

def get_and_classify_audio(audio_recorder, model_runner, media_player, motor_driver):
    
    print("Getting audio sample...")
    start = time.time()
    # Get the newest sample
    has_new_audio = audio_recorder.__get_recording__()
    end = time.time()
    
    print("Audio sampling in {} seconds.".format(end - start))
    
    # If success, run through the model and play music or stop
    if has_new_audio:
        output_file = audio_recorder.__get_output_name__()
        
        print("Pre-processing audio sample...")
        start = time.time()
        
        # Get the spectrogram image of the newest audio signal
        spectrogram = preprocess_and_create_spectrogram(output_file, noise_reps=5)
        
        end = time.time()
        
        print("Pre-processing in {} seconds.".format(end - start))
        
        print("Evaluating with model...")
        start = time.time()
        
        # Getting the class label
        class_label = model_runner.__evaluate_class__(spectrogram)
        end = time.time()
        
        print("Model evaluated in {} seconds.".format(end - start))
        
        if media_player == None:
            return
        
        print("Class label: ", class_label)
        if class_label == 0:
            media_player.stop()
            motor_driver.__stop_motor__()
        elif class_label == 1 and not media_player.is_playing():
            media_player.play()
            motor_driver.__start_motor__()
            
        print()

print("Setting everything up...")

start = time.time()

# Setting up main objects that control execution
audio_recorder = AudioRecorder()
audio_recorder.__start_audio__()

motor_driver = MotorDriver()
model_runner = ModelRunner()

media_player = vlc.MediaPlayer(LULLABY_FILE)
old_volume = media_player.audio_get_volume()
media_player.audio_set_volume(100)

end = time.time()

print("Start-up in {} seconds".format(end - start))

finish_time = datetime.datetime.now() + datetime.timedelta(seconds=30)

while datetime.datetime.now() < finish_time:
    # Actually records the audio and runs through the model
    get_and_classify_audio(audio_recorder, model_runner, media_player, motor_driver)

# Turn everything off when you are done
audio_recorder.__stop_audio__()
motor_driver.__stop_motor__()

media_player.audio_set_volume(old_volume)
media_player.stop()

print("Done!")
