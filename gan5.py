from IPython.display import clear_output
from ipywidgets import interact, IntSlider

import os
import os.path
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm

#parameters for data, eventually change to just be drums
n_trakcs = 5 
n_pitches = 72
lowest_pitch = 24
n_samples_per_song = 8
n_measures = 4
beat_resolution = 4

programs = [0, 0, 25, 33, 48]
is_drums = [True, False, False, False, False]
track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
tempo = 100

#parameters for training
batch_size = 16
latent_dim = 128
n_steps = 20000

#parameters for sampling
sample_interval = 100
n_samples = 4

measure_resolution = 4 * beat_resolution
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)

dataset_root = Path("C:/Users/rck67/GSET/GAN/lpd_5/lpd_5_cleansed")
id_list = []

for path in os.listdir("C:/Users/rck67/GSET/GAN/amg"):
	filepath = os.path.join("C:/Users/rck67/GSET/GAN/amg", path)
	if os.path.isfile(filepath):
		with open(filepath) as f:
			id_list.extend([line.rstrip() for line in f])
id_list = list(set(id_list))

def msd_id_to_dirs(msd_id):
	return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

data = []

for msd_id in tqdm(id_list):
	#load multitrack 
	song_dir = dataset_root / msd_id_to_dirs(msd_id)
	multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
	#binarize
	multitrack.binarize()
	#downsample
	multitrack.set_resolution(beat_resolution)
	#stack pianoroll
	pianoroll = (multitrack.stack() > 0)
	#target pitch raneg only
	pianoroll = pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches]
	#calc total measures
	n_total_measures = multitrack.get_max_length() // measure_resolution
	candidate = n_total_measures - n_measures
	target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
	#select rand num of phrases from pianoroll
	for idx in np.random.choice(candidate, target_n_samples, False):
		start = idx * measure_resolution
		end = (idx + n_measures) * measure_resolution
		#skip samples with too few notes
		if (pianoroll.sum(axis=(1, 2)) < 10).any():
			continue
		data.append(pianoroll[:, start:end])
#stack pianoroll segments into array
random.shuffle(data)
data = np.stack(data)
print(f"collected {len(data)} samples from {len(id_list)} songs")
print(f"data shape : {data.shape}")

