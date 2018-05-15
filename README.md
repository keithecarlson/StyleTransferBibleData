#StyleTransferBibleData

The code and data here are associated with "Evaluating Prose Style Transfer with the Bible", currently under review.

The training, testing, and development data files are provided with the target style tags at the beginning of each source line as they were used by the seq2seq model in our paper.  This may be useful information for other systems, but versions of these files without the tags will be created by following the instructions below.  

All code was run in a linux virtual environment with tensorflow 1.0.1, seq2seq(https://github.com/google/seq2seq), and Moses(http://www.statmt.org/moses/) all installed and working.   We also use subword-nmt (https://github.com/rsennrich/subword-nmt), but the code is replicated within this repository. Run all commands from within the Code directory unless otherwise noted. 

#Preparation
We need to do a few things before get into the experiments.

First, upload the modified and included experiment.py file into seq2seq/seq2seq/contrib.  This will allow us to run evaluation on all checkpoints of a model.

Specify the directory that holds your moses installation:
	
	MOSES_DIR=~/mosesdecoder/

And also specify the directory that you have installed this repository into:

	REP_DIR=~/StyleTransferBibleData/

Then, combine the public->public training data files back together:  
```
cat $REP_DIR/Data/Samples/publicVersions/Train1.tgt $REP_DIR/Data/Samples/publicVersions/Train2.tgt $REP_DIR/Data/Samples/publicVersions/Train3.tgt $REP_DIR/Data/Samples/publicVersions/Train4.tgt > ~$REP_DIR/Data/Samples/publicVersions/Train.tgt 

cat $REP_DIR/Data/Samples/publicVersions/Train1.sourc $REP_DIR/Data/Samples/publicVersions/Train2.sourc $REP_DIR/Data/Samples/publicVersions/Train3.sourc $REP_DIR/Data/Samples/publicVersions/Train4.sourc > $REP_DIR/Data/Samples/publicVersions/Train.sourc 

rm $REP_DIR/Data/Samples/publicVersions/Train1.tgt
rm $REP_DIR/Data/Samples/publicVersions/Train2.tgt
rm $REP_DIR/Data/Samples/publicVersions/Train3.tgt
rm $REP_DIR/Data/Samples/publicVersions/Train4.tgt

rm $REP_DIR/Data/Samples/publicVersions/Train1.sourc
rm $REP_DIR/Data/Samples/publicVersions/Train2.sourc
rm $REP_DIR/Data/Samples/publicVersions/Train3.sourc
rm $REP_DIR/Data/Samples/publicVersions/Train4.sourc
```


We need to create versions of the files which are simply tokenized using nltk and files with the BPE vocabulary applied to them which we will use for Seq2Seq.  From within the Code directory:

	python2 simpleTokenizer.py

	mkdir ../Data/BPESamples

	python massApplyBPE.py ../Data/Samples/ ../Data/BPESamples/ ../Data/simpleTokenBPECodes30000.txt


Next we will create versions of the sourc files which don't have the target version tag prepended.  These will be used for training Moses and evaluating all results:

```	
python removeTags.py
```

For Moses we also need to remove special characters from some files to prevent it from breaking during the run:

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/publicVersionsToBBE/unTaggedDev.sourc > $REP_DIR/Data/Samples/publicVersionsToBBE/unTaggedDevClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/publicVersionsToBBE/unTaggedTest.sourc > $REP_DIR/Data/Samples/publicVersionsToBBE/unTaggedTestClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/YLTToBBE/unTaggedDev.sourc > $REP_DIR/Data/Samples/YLTToBBE/unTaggedDevClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/YLTToBBE/unTaggedTest.sourc > $REP_DIR/Data/Samples/YLTToBBE/unTaggedTestClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/BBEToASV/unTaggedDev.sourc > $REP_DIR/Data/Samples/BBEToASV/unTaggedDevClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/BBEToASV/unTaggedTest.sourc > $REP_DIR/Data/Samples/BBEToASV/unTaggedTestClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/KJVToASV/unTaggedDev.sourc > $REP_DIR/Data/Samples/KJVToASV/unTaggedDevClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/KJVToASV/unTaggedTest.sourc > $REP_DIR/Data/Samples/KJVToASV/unTaggedTestClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/publicVersionsToASV/unTaggedDev.sourc > $REP_DIR/Data/Samples/publicVersionsToASV/unTaggedDevClean.sourc

	$MOSES_DIR/scripts/tokenizer/escape-special-chars.perl < $REP_DIR/Data/Samples/publicVersionsToASV/unTaggedTest.sourc > $REP_DIR/Data/Samples/publicVersionsToASV/unTaggedTestClean.sourc


You should now have 4 populated directories within Data.  These should be Bibles, Samples, simpleTokenSamples, and BPESamples.  Each of the Samples directories should contain 6 directories: publicVersions, publicVersionsToBBE, YLTToBBE, KJVToASV, and BBEToASV.  These should all contain the appropriate test, train and dev files for the dataset and processing, this includes unTagged and unTagged..Clean files in the Samples directory.

Now you are ready to begin the various experiments.


#To train a Seq2Seq model:

Make sure we have done all of the correct BPE data set up.  This means we simply tokenize the raw samples, and apply the BPE vocab.
All of these files should be in Data/BPESamples

Set names of directories to use.  DATA_SET should be named the same as one of the directories in Data/BPESamples to match which pairing's data you want to use as training and development:

	REP_DIR=~/BibleStyleTransfer/
	DATA_SET=publicVersions
	MODEL_DIR=$REP_DIR/Models/${DATA_SET}Seq2Seq
	mkdir -p $MODEL_DIR

Run model on the chosen data.  You may want to modify train_steps and save_checkpoints_steps below for different data sets. :

	python -m bin.train \
	  --config_paths="
	      $REP_DIR/Code/Seq2SeqConfigs/config.yml,
	      $REP_DIR/Code/Seq2SeqConfigs/trainSeq2Seq.yml,
	      $REP_DIR/Code/Seq2SeqConfigs/text_metrics_bpe.yml" \
	  --model_params "
	      vocab_source: $REP_DIR/Data/simpleTokenBPEVocab30000Clean.txt
	      vocab_target: $REP_DIR/Data/simpleTokenBPEVocab30000Clean.txt" \
	  --input_pipeline_train "
	    class: ParallelTextInputPipeline
	    params:
	      shuffle: True
	      source_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Train.sourc
	      target_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Train.tgt" \
	  --input_pipeline_dev "
	    class: ParallelTextInputPipeline
	    params:
	      source_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Dev.sourc
	      target_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Dev.tgt" \
	  --batch_size 64 \
	  --train_steps 200000 \
	  --keep_checkpoint_max 0 \
	  --save_checkpoints_steps 5000 \
	  --schedule train \
	  --output_dir $MODEL_DIR   


Evaluate all saved model checkpoints on a sample of the dev data:

	python -m bin.train \
	  --config_paths="
	      $REP_DIR/Code/Seq2SeqConfigs/config.yml,
	      $REP_DIR/Code/Seq2SeqConfigs/trainSeq2Seq.yml,
	      $REP_DIR/Code/Seq2SeqConfigs/text_metrics_bpe.yml" \
	  --model_params "
	      vocab_source: $REP_DIR/Data/simpleTokenBPEVocab30000Clean.txt
	      vocab_target: $REP_DIR/Data/simpleTokenBPEVocab30000Clean.txt" \
	  --input_pipeline_train "
	    class: ParallelTextInputPipeline
	    params:
	      shuffle: True
	      source_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Train.sourc
	      target_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Train.tgt" \
	  --input_pipeline_dev "
	    class: ParallelTextInputPipeline
	    params:
	      source_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Dev.sourc
	      target_files:
	        - $REP_DIR/Data/BPESamples/${DATA_SET}/Dev.tgt" \
	  --batch_size 64 \
	  --schedule evalAllCheckpoints \
	  --output_dir $MODEL_DIR   


once all checkpoints have been evaluated look at them using tensorboard.

	tensorboard --logdir $MODEL_DIR

We use the loss on eval_all and identify the lowest value as our best checkpoint.  If the lowest value is the final one then you should increase train_steps and continue training.
	
	export BEST_CKPT=130002

Decode using this best checkpoint.  TEST_SET should again be named the same as one of the directories in Data/BPESamples, this time to indicate which data you want to decode:

	export PRED_DIR=$MODEL_DIR/decode$BEST_CKPT
	export TEST_SET=YLTToBBE
	mkdir -p ${PRED_DIR}


	python -m bin.infer \
	  --tasks "
	    - class: DecodeText"\
	  --model_dir $MODEL_DIR \
	  --checkpoint_path ${MODEL_DIR}/model.ckpt-$BEST_CKPT \
	  --model_params "
	    inference.beam_search.beam_width: 10" \
	  --input_pipeline "
	    class: ParallelTextInputPipeline
	    params:
	      source_files:
	       - $REP_DIR/Data/BPESamples/${TEST_SET}/Test.sourc" \
	  > ${PRED_DIR}/${TEST_SET}TestPredictions.txt

This should create a file in your $PRED_DIR containing the decoded output. 
Since the model was trained on the files with the BPE vocab applied, change the results of decoding back to normal English:

	cp ${PRED_DIR}/${TEST_SET}TestPredictions.txt ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt

	sed -i 's/@@ //g' ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt

Also we will undo the NLTK tokenization that we did at the beginning.

	python deTokenizeFile.py -s ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt -o ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt

Evaluate the performance using BLEU and PINC:

	perl $REP_DIR/Code/multi-bleu.perl $REP_DIR/Data/Samples/${TEST_SET}/Test.tgt <  ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt

	perl $REP_DIR/Code/PINC.perl $REP_DIR/Data/Samples/${TEST_SET}/unTaggedTest.sourc <  ${PRED_DIR}/${TEST_SET}TestPredictionsClean.txt

You can repeat the decoding, cleaning, and scoring for each test data you would like to evaluate this model on.