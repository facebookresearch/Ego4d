#! /bin/bash


if [ $# -lt 2 ]; then
    echo "Missing output-dir and stage args"
    exit 1
fi

# -------------------
# Modify according to your paths
PATH=/home/$USER/sctk/SCTK-master/bin:$PATH # sclite
GLM=english.glm # kaldi_en.glm # Kaldi GitHub
audioDir=../ego4d_data/wavs_16000
videoDir=../ego4d_data/clips
ego4d=../ego4d_data
# -------------------

outdir=$1
stage=$2
mkdir -p $outdir

if [[ $stage -le 0 ]]; then
    if [ -d $audioDir ]; then
        echo "Audio dir already exists. Exiting"
        echo "If you want to re-run audio extraction, rm $audioDir"
        exit 1;
    fi
    echo "Running audio extraction from videos..."
    extract_wav.sh $videoDir $audioDir
fi


if [[ $stage -le 1 ]]; then
    for subset in "train" "val" "test"; do
        echo "Preparing reference for $subset subset..."
        # Reference TRN
        splitFile=$ego4d/annotations/av_${subset}.json
        python extract_transcripts.py $splitFile $outdir/ref.$subset.csv $outdir/ref.$subset.trn
        # GLM norm the trn
        csrfilt.sh -s -i trn $GLM \
                   < $outdir/ref.$subset.trn \
                   > $outdir/ref.$subset.filt.trn
    done
fi


if [[ $stage -le 2 ]]; then
    for vadType in "system" "oracle"; do
        labDir=../vad_${vadType}/labs

        echo $labDir
        if [ ! -d $labDir ]; then
            echo "VAD dir does not exist, either run VAD"
            echo "or use a fixed segment length for decoding"
            echo "by setting the --segment_length option "
            echo "in the call for decode_audio.py"
            exit 1;
        fi

        for subset in "val" "test"; do
            echo "Processsing $subset with VAD type $vadType ..."

            splitFile=$outdir/ref.$subset.csv
            # Hypothesis TRN
            # Obtain the ASR decodes
            # This will take a while
            python decode_audio.py $splitFile $audioDir $labDir \
                   $outdir/hyp.$subset.${vadType}.trn

            # Normalize transcript
            csrfilt.sh -s -i trn $GLM \
                       < $outdir/hyp.$subset.${vadType}.trn \
                       > $outdir/hyp.$subset.${vadType}.filt.trn

            # Score the decoding output
            sclite -r $outdir/ref.$subset.filt.trn trn \
                   -h $outdir/hyp.$subset.${vadType}.filt.trn trn \
                   -i rm -o pra -o dtl -o sum \
                   -n score_${subset}_vad_${vadType}
        done
    done
fi
