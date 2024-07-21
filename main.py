import zipfile
import os
import glob
from music21 import converter, instrument, note, chord, stream, tempo, dynamics
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Input, Bidirectional, Conv1D, MaxPooling1D, Flatten, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from midi2audio import FluidSynth
from pydub import AudioSegment
import random

# Ensure GPU is used if available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Paths for the zip file and extracted MIDI files
zip_path = 'archive (4).zip'
extract_path = '/sample_data/midi_files'

# Extract the zip file
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print("Extraction completed.")
else:
    print("The specified zip file does not exist.")

# Function to get notes from MIDI files
def get_notes(midi_path):
    notes = []
    for file in glob.glob(f"{midi_path}/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:  # file has instrument parts
            for part in parts.parts:
                notes_to_parse = part.recurse()
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append((str(element.pitch), element.offset, type(part.getInstrument()), element.volume.velocity))
                    elif isinstance(element, chord.Chord):
                        notes.append(('.'.join(str(n) for n in element.normalOrder), element.offset, type(part.getInstrument()), element.volume.velocity))
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append((str(element.pitch), element.offset, instrument.Piano, element.volume.velocity))
                elif isinstance(element, chord.Chord):
                    notes.append(('.'.join(str(n) for n in element.normalOrder), element.offset, instrument.Piano, element.volume.velocity))
    return notes

# Extract notes
midi_path = extract_path
notes = get_notes(midi_path)

# Get unique notes and their mappings
unique_notes = sorted(set([n[0] for n in notes]))
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
int_to_note = dict((number, note) for number, note in enumerate(unique_notes))

# Prepare sequences for the Neural Network
sequence_length = 100
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char[0]] for char in sequence_in])
    network_output.append(note_to_int[sequence_out[0]])

n_patterns = len(network_input)

# Reshape and normalize the input
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(unique_notes))

network_output = to_categorical(network_output)

# Define the model
input_shape = (network_input.shape[1], network_input.shape[2])

def create_model():
    model_input = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Attention()([x, x])

    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = LSTM(512)(x)
    x = Dropout(0.3)(x)

    x = Dense(256)(x)
    x = Dropout(0.3)(x)
    output = Dense(len(unique_notes), activation='softmax')(x)

    model = Model(model_input, output)

    return model

# Build multiple models and compile
models = [create_model() for _ in range(3)]
for model in models:
    model.compile(loss='categorical_crossentropy', optimizer=Adam())

print("Models compiled successfully.")

# Train the models
epochs = 200
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    model.fit(network_input, network_output, epochs=epochs, batch_size=64, verbose=2)
    model.save(f'/mnt/data/lofi_music_generator_{i+1}.h5')

# Load the trained models
models = [load_model(f'/mnt/data/lofi_music_generator_{i+1}.h5') for i in range(3)]

# Generate music using the ensemble
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]

generated_notes = []

for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(unique_notes))

    # Ensemble prediction
    predictions = [model.predict(prediction_input, verbose=0) for model in models]
    prediction = np.mean(predictions, axis=0)

    index = np.argmax(prediction)
    result = int_to_note[index]
    generated_notes.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]

# Convert generated notes to MIDI
offset = 0
output_notes = []

# Instruments for lofi style
instruments = [instrument.AcousticGuitar(), instrument.ElectricPiano(), instrument.Flute(), instrument.Bass(), instrument.Percussion()]

for pattern in generated_notes:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = random.choice(instruments)
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = random.choice(instruments)
        output_notes.append(new_note)

    offset += 1.0  # Longer note duration for lofi

midi_stream = stream.Stream(output_notes)
midi_file_path = '/mnt/data/output_lofi_music.mid'
midi_stream.write('midi', fp=midi_file_path)

print("Music generation completed. MIDI file saved.")

# Convert MIDI to WAV using FluidSynth
wav_file_path = '/mnt/data/output_lofi_music.wav'
fs = FluidSynth()
fs.midi_to_audio(midi_file_path, wav_file_path)

# Apply post-processing to the WAV file using pydub
audio = AudioSegment.from_wav(wav_file_path)

# Apply reverb
audio = audio.overlay(audio, delay=50)

# Apply equalization
audio = audio.low_pass_filter(300)

# Apply compression
audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0)

# Convert WAV to MP3
mp3_file_path = '/mnt/data/output_lofi_music.mp3'
audio.export(mp3_file_path, format="mp3")

print("Conversion to MP3 completed. Check 'output_lofi_music.mp3' for the generated audio file.")
