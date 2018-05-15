
from nltk.tokenize import word_tokenize
import os

def tokenizeFile(inputFileName, outputFileName):
	lines = open(inputFileName,'r').readlines()
	outFile = open(outputFileName,'w')

	for line in lines:
		line = unicode(line,'utf-8')
		parts = word_tokenize(line)
		if len(parts)>2:
			if parts[0] == '<' and parts[2] == '>':
				parts[2] = '<'+parts[1]+'>'
				parts = parts[2:]
		words = ' '.join(parts)
		
		outFile.write( words.encode('utf8') + '\n')
		



def recursiveSimpleTokenize(rawDir, tokenDir):
	if not os.path.exists(tokenDir):
		os.mkdir(tokenDir)

	for root, subdirs, files in os.walk(rawDir):
		for subdir in subdirs:
			dir_path = os.path.join(root, subdir)
			
			dir_path = dir_path.replace(rawDir,tokenDir)
			if not os.path.exists(dir_path):
				os.mkdir(dir_path)
		for file in files:
			read_file_path = os.path.join(root, file)
			write_file_path = os.path.join(root, file).replace(rawDir,tokenDir)
			if not os.path.exists(write_file_path):
					tokenizeFile(read_file_path, write_file_path)


if __name__ == '__main__': 
	sampleOutDir = os.path.join('..','Data','Samples')

	simpleTokensOutDir = os.path.join('..','Data','simpleTokenSamples')

	if not os.path.exists(simpleTokensOutDir):
		os.mkdir(simpleTokensOutDir)
		
	recursiveSimpleTokenize(sampleOutDir, simpleTokensOutDir)
