import os
import numpy as np
import soundfile
from espnet2.bin.asr_inference import Speech2Text
from asr_utils import get_list_of_clips


def transcribe(speech2text, speech, rate, vad_file):
    text_list = []
    with open(vad_file, 'r') as f:
        for line in f:
            tb, te, sp_no_sp = line.strip().split()
            if sp_no_sp == "speech":
                sample_begin = int(float(tb) * rate)
                sample_end = int(float(te) * rate)
                try:
                    nbests = speech2text(speech[sample_begin:sample_end])
                    text, *_ = nbests[0]
                    text_list.append(text)
                except:
                    continue

    result_text = " ".join(text_list)
    return result_text.strip().lower().replace("<sos/eos>", "")


def transcribe_segmented(speech2text, speech, rate, segment_length=5.0):
    text_list = []
    N = int(np.size(speech))
    Nhop = int(segment_length * rate)

    for sample_begin in np.arange(0, N-Nhop+1, Nhop):
        sample_end = sample_begin + Nhop
        nbests = speech2text(speech[sample_begin:sample_end])
        text, *_ = nbests[0]
        text = text.strip()
        if text != "":
            text_list.append(text)

    result_text = " ".join(text_list)
    return result_text.strip().lower().replace("<sos/eos>", "")


def main(args):
    assert(os.path.exists(args.audio_dir))
    assert(os.path.exists(args.lab_dir))
    print("If no VAD, segment length will be ", args.segment_length)

    speech2text = Speech2Text.from_pretrained(
        "Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave",
        # Decoding parameters are not included in the model file
    maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=20,
        ctc_weight=0.3,
        lm_weight=0.5,
        penalty=0.0,
        nbest=1
    )

    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text

    clip_list = get_list_of_clips(args.split_file)

    with open(args.hyp_trn, 'w') as trn_f:
        ctr = 0
        for clip in clip_list:
            audio_file = os.path.join(args.audio_dir, clip+'.wav')
            vad_file = os.path.join(args.lab_dir, clip+'.lab')

            speech, rate = soundfile.read(audio_file)
            assert(rate == 16000)
            text_list = []
            if os.path.exists(vad_file):
                hyp_text = transcribe(speech2text, speech, rate, vad_file)
            else:
                hyp_text = transcribe_segmented(
                    speech2text, speech, rate, segment_length=args.segment_length
                )
            trn_f.write("{} ({}_1_0_30000)\n".format(hyp_text, clip))
            ctr += 1
            if ctr % 20 == 0:
                print("Processed {} clips".format(ctr))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split_file",
        type=str,
        help="CSV file that contains train/val/test set info"
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory of the extracted 16kHz audio files"
    )
    parser.add_argument(
        "lab_dir",
        type=str,
        help="Directory of the VAD .lab files"
    )
    parser.add_argument(
        "hyp_trn",
        type=str,
        help="Output TRN file that will contain the ASR hypotheses"
    )
    parser.add_argument(
        "--segment_length",
        type=float,
        help="If VAD labs are not available,"
        "you can adjust the segment length for decoding (in seconds)",
        default=10.0
    )
    args = parser.parse_args()
    main(args)
