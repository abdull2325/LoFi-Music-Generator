# LoFi Music Generator

## Overview

This project implements an advanced ensemble model for generating LoFi music using deep learning techniques. It processes MIDI files, trains multiple neural networks, and generates new LoFi compositions, complete with post-processing effects.

## Features

- Extracts musical features from MIDI files
- Implements an ensemble of deep learning models for music generation
- Utilizes convolutional and recurrent neural networks with attention mechanisms
- Generates MIDI files with LoFi-style instrumentation
- Converts generated MIDI to audio with post-processing effects

## Requirements

- Python 3.7+
- TensorFlow 2.x
- music21
- numpy
- midi2audio
- FluidSynth
- pydub

## Installation

1. Clone this repository
2.  2. Install the required packages
3. Ensure FluidSynth is installed on your system.

## Usage

1. Place your MIDI dataset in a zip file named 'archive (4).zip' in the project root.

2. Run the main script
3. The script will extract MIDI files, train the models, generate new music, and save the output as both MIDI and MP3 files.

## Algorithm Details

This project employs a sophisticated approach to music generation:

1. **Data Preparation**:
- Extracts notes and chords from MIDI files
- Converts musical elements to numerical sequences

2. **Model Architecture**:
- Ensemble of multiple models
- Each model includes:
  - Convolutional layers for feature extraction
  - Bidirectional LSTM layers for sequence processing
  - Attention mechanism for focusing on important parts of the sequence
  - Dense layers for final prediction

3. **Training Process**:
- Models are trained on preprocessed musical sequences
- Uses categorical cross-entropy loss and Adam optimizer

4. **Music Generation**:
- Ensemble prediction from multiple trained models
- Generates a sequence of notes and chords

5. **Post-Processing**:
- Converts generated sequence to MIDI
- Applies LoFi-style instrumentation
- Converts MIDI to audio (WAV)
- Applies audio effects (reverb, EQ, compression)
- Exports final result as MP3

## Impact and Contribution

This project contributes to the field of AI-generated music:

1. **Creative Tool**: Provides musicians and producers with a new source of musical ideas and inspiration.

2. **LoFi Genre Exploration**: Specifically targets the popular LoFi genre, exploring its musical characteristics through AI.

3. **Educational Resource**: Serves as a learning tool for those interested in music generation using deep learning.

4. **Customizable Framework**: Can be adapted for generating music in other genres or styles.

5. **Audio Processing Pipeline**: Demonstrates a complete pipeline from MIDI generation to final audio production.

## Future Work

- Implement more diverse musical features (e.g., rhythm, dynamics)
- Explore additional deep learning architectures
- Develop a user interface for easier interaction with the model
- Incorporate real-time generation capabilities

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- The creators and contributors of the music21 library
- TensorFlow and Keras teams
- FluidSynth and pydub developers
- The LoFi music community for inspiration
   
