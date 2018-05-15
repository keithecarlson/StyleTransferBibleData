import os

def removeTagsFromLines(fileNameIn,fileNameOut):
	file = open(fileNameIn,'r')
	lines =file.readlines()
	file.close()
	outFile = open(fileNameOut,'w')
	for line in lines:
		if line[0]=='<':
			line = line[line.find(' ')+1:]
		outFile.write(line)
	outFile.close()

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToASV','Test.sourc'),os.path.join('..','Data','Samples','publicVersionsToASV','unTaggedTest.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToASV','Train.sourc'),os.path.join('..','Data','Samples','publicVersionsToASV','unTaggedTrain.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToASV','Dev.sourc'),os.path.join('..','Data','Samples','publicVersionsToASV','unTaggedDev.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToBBE','Test.sourc'),os.path.join('..','Data','Samples','publicVersionsToBBE','unTaggedTest.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToBBE','Train.sourc'),os.path.join('..','Data','Samples','publicVersionsToBBE','unTaggedTrain.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','publicVersionsToBBE','Dev.sourc'),os.path.join('..','Data','Samples','publicVersionsToBBE','unTaggedDev.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','KJVToASV','Test.sourc'),os.path.join('..','Data','Samples','KJVToASV','unTaggedTest.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','KJVToASV','Train.sourc'),os.path.join('..','Data','Samples','KJVToASV','unTaggedTrain.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','KJVToASV','Dev.sourc'),os.path.join('..','Data','Samples','KJVToASV','unTaggedDev.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','BBEToASV','Test.sourc'),os.path.join('..','Data','Samples','BBEToASV','unTaggedTest.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','BBEToASV','Train.sourc'),os.path.join('..','Data','Samples','BBEToASV','unTaggedTrain.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','BBEToASV','Dev.sourc'),os.path.join('..','Data','Samples','BBEToASV','unTaggedDev.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','YLTToBBE','Test.sourc'),os.path.join('..','Data','Samples','YLTToBBE','unTaggedTest.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','YLTToBBE','Train.sourc'),os.path.join('..','Data','Samples','YLTToBBE','unTaggedTrain.sourc'))

removeTagsFromLines(os.path.join('..','Data','Samples','YLTToBBE','Dev.sourc'),os.path.join('..','Data','Samples','YLTToBBE','unTaggedDev.sourc'))