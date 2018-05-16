from optparse import OptionParser
from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()
parser = OptionParser()
parser.add_option("-s",  dest="sourceFilename",  metavar="SOURCEFILE")
parser.add_option("-o",  dest="outputFilename",  metavar="OUTPUTFILE")

(options, args) = parser.parse_args()


with open(options.sourceFilename,'r') as sourceFile:
	lines = sourceFile.readlines()

with open(options.outputFilename,'w') as outFile:
	for line in lines:
		parts = line.split(' ')
		cleanLine = detokenizer.detokenize(parts, return_str=True)
		outFile.write(cleanLine + '\n')
