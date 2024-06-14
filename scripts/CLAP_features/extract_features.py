import os
import numpy as np 
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
import torch
from transformers import ClapModel, ClapProcessor, AutoFeatureExtractor


def ospif(file):
	return os.path.isfile(file)


def ospid(dir_):
	return os.path.isdir(dir_)


def pkl_dmp(obj, fp):
	with open(fp, "wb") as fo:
		pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)

        
def pkl_ld(fp):
	with open(fp, "rb") as fi:
		pkl_content = pickle.load(fi)
	return pkl_content


def json_ld(fp):
	with open(fp, "r") as fi:
		json_content = json.load(fi)
	return json_content


def json_dmp(obj, fp, indent=None):
	with open(fp, "w") as fo:
		if indent is None:
			json.dump(obj, fo)
		else:
			assert isinstance(indent, int)
			json.dump(obj, fo, indent=indent)


INT32_MAX = 2_147_483_648


TAKES_ROOT_DR = f"/datasets01/egoexo4d/v2/takes"
assert ospid(TAKES_ROOT_DR)

DUMP_DR = "/large_experiments/eht/research/egoexo/clap_features"
assert ospid(DUMP_DR)


model = ClapModel.from_pretrained("laion/larger_clap_general").to(0)
processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/larger_clap_general")

lst_alNmChnks = []
for tk_idx, tk_drn in enumerate(tqdm(os.listdir(TAKES_ROOT_DR))):
	if ospif(f"{DUMP_DR}/{tk_drn}.pt"):
		continue
	
	audio_fps = glob(f"{TAKES_ROOT_DR}/{tk_drn}/audio/aria*.wav")
	assert len(audio_fps) in [0, 1], print(audio_fps)
	if len(audio_fps) == 0:
		continue
	audio_fp = audio_fps[0]
	assert ospif(audio_fp), print(audio_fp)

	sr, audio_data = wavfile.read(audio_fp)
	assert sr == 48000
	assert audio_data.dtype == np.int32

	audioData_ch1 = audio_data[:, 1]
	audioData_ch1 = (audioData_ch1.astype(np.float64) / INT32_MAX).astype(np.float32)

	if len(audioData_ch1) % sr == 0:
		num_chunks = len(audioData_ch1) // sr
	else:
		num_chunks = (len(audioData_ch1) // sr) + 1

	lst_alNmChnks.append(num_chunks)

	# print("1: ", len(audioData_ch1) / sr, num_chunks)

	lst_feats = []
	for i in tqdm(range(num_chunks)):
		audio_chunk = audioData_ch1[i * sr: (i + 1) * sr]
		audioChunk_inputs = feature_extractor(audio_chunk, return_tensors="pt", sampling_rate=48000).to(0)
		audioChunk_features = model.get_audio_features(**audioChunk_inputs)[0].detach().cpu()
		# print(audioChunk_features.shape, audioChunk_features.dtype, audioChunk_features.device)
		assert len(audioChunk_features) == 512
		lst_feats.append(audioChunk_features)

		# break

	feats = torch.stack(lst_feats)
	torch.save(feats, f"{DUMP_DR}/{tk_drn}.pt")

	# break

	tk_idx += 1
	# if tk_idx == 5:
	# 	break


print(np.min(lst_alNmChnks), 
	  np.max(lst_alNmChnks),
	  np.mean(lst_alNmChnks),
	  np.std(lst_alNmChnks),
	  np.sum(lst_alNmChnks))





