from pydub import AudioSegment
from pydub.playback import play
name = 'finish'
play(AudioSegment.from_wav(name))